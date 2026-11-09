//! WAN benchmark for the WSS video path.
//!
//! Runs the real session loop (`run_session_cpu`), the real OpenH264 encoder
//! and the real WSS send path (`transport::ws::raw_ws_session`: same bounded
//! queue and I/O loop as HTTPS, minus TLS), through a userspace bottleneck
//! link, into a client that decodes every frame with OpenH264 and reads a
//! timestamp marker drawn into the content — the measurement method of
//! voxl's `bench/` (marker-to-decode latency, unique frames per second).
//!
//! Modes:
//! - `ack`   — the client sends `VideoAck` after decoding (current web and
//!   native clients): ack-based flow control, delay-based rate control,
//!   no periodic keyframes.
//! - `noack` — the client never acks: the server's pre-ack behaviour
//!   (ping-based ABR, send-time congestion tracker, 2 s keyframes).
//!
//! Build with `--cfg phantom_baseline` against a tree without `VideoAck` to
//! measure an older revision (only `noack` is available there).
//!
//! ```text
//! cargo run --release -p phantom-server --no-default-features --example wan_bench -- \
//!     --secs 20 --profiles lan,20/20,10/40,5/60,2/100 --workloads scroll,anim,window
//! ```
//!
//! Output: one line per run, plus a JSON line (`JSON {...}`) for scripts.
#![allow(unexpected_cfgs)]

use anyhow::Result;
use phantom_core::capture::FrameCapture;
use phantom_core::encode::VideoCodec;
use phantom_core::frame::{Frame, PixelFormat};
use phantom_core::input::InputEvent;
use phantom_core::protocol::Message;
use phantom_core::tile::TileDiffer;
use phantom_server::encode::h264::OpenH264Encoder;
use phantom_server::session::{run_session_cpu, InputForwarder, SessionConfig};
use phantom_server::transport::ws::raw_ws_session;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::io::{ErrorKind, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// ── Configuration ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Link {
    /// Bottleneck rate, bits/s; 0 = unlimited.
    rate_bps: u64,
    /// Round-trip propagation delay (split evenly per direction).
    rtt_ms: u64,
    /// Bottleneck queue, in ms of link time (bufferbloat depth).
    queue_ms: u64,
}

impl Link {
    fn parse(s: &str, queue_ms: u64) -> Link {
        if s == "lan" {
            return Link {
                rate_bps: 0,
                rtt_ms: 0,
                queue_ms,
            };
        }
        let (mbps, rtt) = s.split_once('/').expect("profile is lan or MBPS/RTT_MS");
        Link {
            rate_bps: (mbps.parse::<f64>().expect("mbps") * 1e6) as u64,
            rtt_ms: rtt.parse().expect("rtt ms"),
            queue_ms,
        }
    }

    fn name(&self) -> String {
        if self.rate_bps == 0 && self.rtt_ms == 0 {
            "LAN".into()
        } else {
            format!("{}Mbps/{}ms", self.rate_bps as f64 / 1e6, self.rtt_ms)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Workload {
    /// Smooth-scrolling a text-heavy page, full screen.
    Scroll,
    /// Full-screen animation (every pixel changes every frame).
    Anim,
    /// Static desktop with a 960x540 animated window (video playing).
    Window,
}

impl Workload {
    fn parse(s: &str) -> Workload {
        match s {
            "scroll" => Workload::Scroll,
            "anim" => Workload::Anim,
            "window" => Workload::Window,
            _ => panic!("workload is scroll, anim or window"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Ack,
    NoAck,
}

struct Config {
    secs: f64,
    warmup: f64,
    width: u32,
    height: u32,
    fps: u32,
    src_fps: f64,
    bitrate: u32,
    encoder: String,
    reps: u32,
    links: Vec<Link>,
    workloads: Vec<Workload>,
    modes: Vec<Mode>,
}

fn parse_args() -> Config {
    let mut c = Config {
        secs: 20.0,
        warmup: 3.0,
        width: 1920,
        height: 1080,
        fps: 30,
        src_fps: 60.0,
        bitrate: 5000,
        encoder: "openh264".into(),
        reps: 1,
        links: vec![],
        workloads: vec![Workload::Scroll, Workload::Anim, Workload::Window],
        modes: if cfg!(phantom_baseline) {
            vec![Mode::NoAck]
        } else {
            vec![Mode::NoAck, Mode::Ack]
        },
    };
    let mut profiles = "lan,20/20,10/40,5/60,2/100".to_string();
    let mut queue_ms = 500u64;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let v = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--secs" => c.secs = v.parse().unwrap(),
            "--warmup" => c.warmup = v.parse().unwrap(),
            "--res" => {
                let (w, h) = v.split_once('x').unwrap();
                c.width = w.parse().unwrap();
                c.height = h.parse().unwrap();
            }
            "--fps" => c.fps = v.parse().unwrap(),
            "--src-fps" => c.src_fps = v.parse().unwrap(),
            "--bitrate" => c.bitrate = v.parse().unwrap(),
            "--encoder" => c.encoder = v,
            "--reps" => c.reps = v.parse().unwrap(),
            "--queue-ms" => queue_ms = v.parse().unwrap(),
            "--profiles" => profiles = v,
            "--workloads" => c.workloads = v.split(',').map(Workload::parse).collect(),
            "--modes" => {
                c.modes = v
                    .split(',')
                    .map(|m| match m {
                        "ack" => Mode::Ack,
                        "noack" => Mode::NoAck,
                        _ => panic!("mode is ack or noack"),
                    })
                    .collect()
            }
            other => panic!("unknown flag {other}"),
        }
        i += 2;
    }
    c.links = profiles
        .split(',')
        .map(|p| Link::parse(p, queue_ms))
        .collect();
    assert!(c.width >= 1024, "marker needs at least 1024 px width");
    c
}

// ── Synthetic screen with a timestamp marker ────────────────────────────────

const MARKER_CELL: usize = 32;
const MARKER_BITS: usize = 32;

/// Produces a new frame whenever the 60 Hz source content changes (like a
/// damage-driven capture), with the time the content was drawn encoded in
/// the top-left corner: row 0 = 32 bits of ms since the bench epoch, row 1 =
/// their complement (a validity check).
struct SynthCapture {
    w: usize,
    h: usize,
    workload: Workload,
    epoch: Instant,
    src_fps: f64,
    last_idx: Option<u64>,
    texture: Vec<u8>,
    tex_h: usize,
    sin: Vec<u8>,
}

impl SynthCapture {
    fn new(w: u32, h: u32, workload: Workload, epoch: Instant, src_fps: f64) -> Self {
        let (w, h) = (w as usize, h as usize);
        let tex_h = h * 4;
        let mut texture = vec![245u8; w * tex_h];
        let mut rng = 0x2545_f491u32;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng
        };
        // Text-like page: 22 px lines of dark "words", an image block now and then.
        for line in 0..tex_h / 22 {
            let y0 = line * 22 + 5;
            if next() % 9 == 0 {
                let x0 = (next() as usize % (w / 2)) + 40;
                let bw = 200 + next() as usize % 300;
                for y in y0..(y0 + 120).min(tex_h) {
                    for x in x0..(x0 + bw).min(w) {
                        texture[y * w + x] = ((x * 7 + y * 3) % 200) as u8 + 30;
                    }
                }
                continue;
            }
            let mut x = 40;
            while x < w - 80 {
                let ww = 12 + next() as usize % 70;
                for y in y0..(y0 + 12).min(tex_h) {
                    for xx in x..(x + ww).min(w - 40) {
                        texture[y * w + xx] = 30 + (next() % 40) as u8;
                    }
                }
                x += ww + 8;
            }
        }
        let sin = (0..1024)
            .map(|i| ((i as f64 * std::f64::consts::TAU / 1024.0).sin() * 63.0 + 64.0) as u8)
            .collect();
        Self {
            w,
            h,
            workload,
            epoch,
            src_fps,
            last_idx: None,
            texture,
            tex_h,
            sin,
        }
    }

    fn render_scroll(&self, out: &mut [u8], idx: u64, x0: usize, y0: usize, w: usize, h: usize) {
        let span = self.tex_h - h;
        let off = (idx as usize * 12) % span;
        for y in 0..h {
            let src = &self.texture[(off + y) * self.w + x0..(off + y) * self.w + x0 + w];
            let row = &mut out[((y0 + y) * self.w + x0) * 4..((y0 + y) * self.w + x0 + w) * 4];
            for (px, &g) in row.chunks_exact_mut(4).zip(src) {
                px[0] = g;
                px[1] = g;
                px[2] = g;
                px[3] = 255;
            }
        }
    }

    fn render_anim(&self, out: &mut [u8], idx: u64, x0: usize, y0: usize, w: usize, h: usize) {
        let t = idx as usize;
        let s = &self.sin;
        for y in 0..h {
            let sy = s[(y * 3 + t * 7) & 1023] as usize;
            let row = &mut out[((y0 + y) * self.w + x0) * 4..((y0 + y) * self.w + x0 + w) * 4];
            for (x, px) in row.chunks_exact_mut(4).enumerate() {
                let a = s[(x * 2 + t * 11) & 1023] as usize;
                let b = s[((x + y) + t * 5) & 1023] as usize;
                px[0] = (a + sy) as u8;
                px[1] = (b + sy / 2) as u8;
                px[2] = (a + b) as u8;
                px[3] = 255;
            }
        }
    }

    fn draw_marker(&self, out: &mut [u8], value: u32) {
        for row in 0..2 {
            let bits = if row == 0 { value } else { !value };
            for bit in 0..MARKER_BITS {
                let on = (bits >> bit) & 1 == 1;
                let v = if on { 235 } else { 16 };
                for y in row * MARKER_CELL..(row + 1) * MARKER_CELL {
                    let base = (y * self.w + bit * MARKER_CELL) * 4;
                    for px in out[base..base + MARKER_CELL * 4].chunks_exact_mut(4) {
                        px[0] = v;
                        px[1] = v;
                        px[2] = v;
                        px[3] = 255;
                    }
                }
            }
        }
    }
}

impl SynthCapture {
    /// Content index whose marker reads `drawn_ms`.
    fn idx_for_ms(&self, drawn_ms: u32) -> u64 {
        (drawn_ms as f64 * self.src_fps / 1000.0).round() as u64
    }

    /// The exact BGRA frame for content index `idx` (deterministic).
    fn render(&self, idx: u64) -> Vec<u8> {
        let drawn_ms = (idx as f64 * 1000.0 / self.src_fps) as u32;
        let (w, h) = (self.w, self.h);
        let mut data = vec![0u8; w * h * 4];
        match self.workload {
            Workload::Scroll => self.render_scroll(&mut data, idx, 0, 0, w, h),
            Workload::Anim => self.render_anim(&mut data, idx, 0, 0, w, h),
            Workload::Window => {
                // Static page (frame 0) with an animated 960x540 window.
                self.render_scroll(&mut data, 0, 0, 0, w, h);
                let (ww, wh) = (960.min(w), 540.min(h - 100));
                self.render_anim(&mut data, idx, (w - ww) / 2, (h - wh) / 2, ww, wh);
            }
        }
        self.draw_marker(&mut data, drawn_ms);
        data
    }
}

impl FrameCapture for SynthCapture {
    fn capture(&mut self) -> Result<Option<Frame>> {
        let idx = (self.epoch.elapsed().as_secs_f64() * self.src_fps) as u64;
        if self.last_idx == Some(idx) {
            return Ok(None);
        }
        self.last_idx = Some(idx);
        Ok(Some(Frame {
            width: self.w as u32,
            height: self.h as u32,
            format: PixelFormat::Bgra8,
            data: self.render(idx),
            timestamp: Instant::now(),
        }))
    }

    fn resolution(&self) -> (u32, u32) {
        (self.w as u32, self.h as u32)
    }
}

fn read_marker(y: &[u8], stride: usize) -> Option<u32> {
    let mut rows = [0u32; 2];
    for (row, out) in rows.iter_mut().enumerate() {
        for bit in 0..MARKER_BITS {
            let cx = bit * MARKER_CELL + MARKER_CELL / 2;
            let cy = row * MARKER_CELL + MARKER_CELL / 2;
            let mut sum = 0u32;
            for yy in cy - 4..cy + 4 {
                for xx in cx - 4..cx + 4 {
                    sum += *y.get(yy * stride + xx)? as u32;
                }
            }
            if sum / 64 > 125 {
                *out |= 1 << bit;
            }
        }
    }
    (rows[0] == !rows[1]).then_some(rows[0])
}

// ── Bottleneck link ─────────────────────────────────────────────────────────

/// One direction of the link: reads from `src`, holds bytes in a FIFO
/// bottleneck queue drained at `rate_bps`, delivers each byte `delay` after
/// it leaves the bottleneck. When the queue holds more than `cap` bytes the
/// reader stops reading, so TCP backpressure reaches the sender — a deep
/// but finite router buffer, not an infinite one.
fn pipe_direction(
    mut src: TcpStream,
    mut dst: TcpStream,
    rate_bps: u64,
    delay: Duration,
    cap: usize,
    stop: Arc<AtomicBool>,
) -> Vec<thread::JoinHandle<()>> {
    type Q = (Mutex<(VecDeque<(Instant, Vec<u8>)>, usize, bool)>, Condvar);
    let q: Arc<Q> = Arc::new((Mutex::new((VecDeque::new(), 0, false)), Condvar::new()));
    let qr = q.clone();
    let stop_r = stop.clone();
    let reader = thread::spawn(move || {
        let mut buf = vec![0u8; 16 * 1024];
        let _ = src.set_read_timeout(Some(Duration::from_millis(50)));
        loop {
            if stop_r.load(Ordering::Relaxed) {
                break;
            }
            {
                let (m, cv) = &*qr;
                let mut g = m.lock().unwrap();
                while g.1 > cap && !stop_r.load(Ordering::Relaxed) {
                    g = cv.wait_timeout(g, Duration::from_millis(20)).unwrap().0;
                }
            }
            match src.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    let (m, cv) = &*qr;
                    let mut g = m.lock().unwrap();
                    // Packetise so serialisation delay is modelled smoothly.
                    for chunk in buf[..n].chunks(1500) {
                        g.0.push_back((Instant::now(), chunk.to_vec()));
                        g.1 += chunk.len();
                    }
                    cv.notify_all();
                }
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
                Err(_) => break,
            }
        }
        let (m, cv) = &*qr;
        m.lock().unwrap().2 = true;
        cv.notify_all();
    });
    let writer = thread::spawn(move || {
        let mut link_free = Instant::now();
        loop {
            let item = {
                let (m, cv) = &*q;
                let mut g = m.lock().unwrap();
                loop {
                    if let Some(it) = g.0.pop_front() {
                        g.1 -= it.1.len();
                        cv.notify_all();
                        break Some(it);
                    }
                    if g.2 || stop.load(Ordering::Relaxed) {
                        break None;
                    }
                    g = cv.wait_timeout(g, Duration::from_millis(20)).unwrap().0;
                }
            };
            let Some((arrived, data)) = item else { break };
            let depart = if rate_bps == 0 {
                arrived
            } else {
                let ser = Duration::from_secs_f64(data.len() as f64 * 8.0 / rate_bps as f64);
                link_free = link_free.max(arrived) + ser;
                link_free
            };
            let deliver = depart + delay;
            let now = Instant::now();
            if deliver > now {
                thread::sleep(deliver - now);
            }
            if dst.write_all(&data).is_err() {
                break;
            }
        }
        let _ = dst.shutdown(Shutdown::Write);
    });
    vec![reader, writer]
}

/// Accepts one client on `listen`, connects to `upstream`, shapes both ways.
fn spawn_link(
    listen: TcpListener,
    upstream: std::net::SocketAddr,
    link: Link,
    stop: Arc<AtomicBool>,
) {
    thread::spawn(move || {
        let (client, _) = listen.accept().expect("link accept");
        let server = TcpStream::connect(upstream).expect("link connect");
        client.set_nodelay(true).ok();
        server.set_nodelay(true).ok();
        let one_way = Duration::from_millis(link.rtt_ms / 2);
        let cap = if link.rate_bps == 0 {
            64 * 1024 * 1024
        } else {
            ((link.rate_bps as f64 / 8.0) * link.queue_ms as f64 / 1000.0) as usize
        };
        // Downstream (server → client) is the bottleneck; upstream only delayed.
        let mut hs = pipe_direction(
            server.try_clone().unwrap(),
            client.try_clone().unwrap(),
            link.rate_bps,
            one_way,
            cap.max(16 * 1024),
            stop.clone(),
        );
        hs.extend(pipe_direction(client, server, 0, one_way, 1 << 20, stop));
        for h in hs {
            let _ = h.join();
        }
    });
}

// ── Measuring client ────────────────────────────────────────────────────────

#[derive(Default)]
struct RunStats {
    latencies_ms: Vec<f64>,
    per_second_unique: BTreeMap<u64, HashSet<u32>>,
    bytes: u64,
    frames: u64,
    keyframes: u64,
    decode_errors: u64,
    marker_misses: u64,
    /// Frames the encoder skipped (empty bitstream) to meet its rate target.
    skipped: u64,
}

fn trace_ms() -> Option<f64> {
    static T: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("WAN_BENCH_TRACE").ok()?.parse().ok())
}

/// HH:MM:SS.mmm (UTC), the clock tracing's log lines use.
fn utc_now() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let s = d.as_secs() % 86400;
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        s / 3600,
        s / 60 % 60,
        s % 60,
        d.subsec_millis()
    )
}

/// (marker ms, decoded luma plane) sent to the quality worker.
type QualitySample = (u32, Vec<u8>);

/// Luma PSNR (dB) of decoded frames against the exact source, computed off
/// the receive thread so it never adds to the measured latency.
fn spawn_quality_worker(
    w: u32,
    h: u32,
    workload: Workload,
    src_fps: f64,
) -> (
    mpsc::SyncSender<QualitySample>,
    thread::JoinHandle<Vec<f64>>,
) {
    let (tx, rx) = mpsc::sync_channel::<QualitySample>(4);
    let join = thread::spawn(move || {
        let src = SynthCapture::new(w, h, workload, Instant::now(), src_fps);
        let (w, h) = (w as usize, h as usize);
        let mut out = Vec::new();
        while let Ok((ms, decoded_y)) = rx.recv() {
            let bgra = src.render(src.idx_for_ms(ms));
            let (y, _, _) = phantom_core::color::bgra_to_yuv420(&bgra, w, h);
            // Skip the marker rows: they are trivially easy to code.
            let start = 2 * MARKER_CELL * w;
            let mse = y[start..]
                .iter()
                .zip(&decoded_y[start..])
                .map(|(&a, &b)| {
                    let d = a as f64 - b as f64;
                    d * d
                })
                .sum::<f64>()
                / (y.len() - start) as f64;
            out.push(if mse == 0.0 {
                99.0
            } else {
                10.0 * (255.0 * 255.0 / mse).log10()
            });
        }
        out
    });
    (tx, join)
}

#[allow(clippy::too_many_arguments)]
fn run_client(
    addr: std::net::SocketAddr,
    mode: Mode,
    epoch: Instant,
    measure_from: Instant,
    until: Instant,
    quality: mpsc::SyncSender<QualitySample>,
    w: usize,
    h: usize,
) -> RunStats {
    let stream = TcpStream::connect(addr).expect("client connect");
    stream.set_nodelay(true).ok();
    stream
        .set_read_timeout(Some(Duration::from_millis(100)))
        .ok();
    let mut ws =
        tungstenite::WebSocket::from_raw_socket(stream, tungstenite::protocol::Role::Client, None);
    let mut decoder = openh264::decoder::Decoder::new().expect("decoder");
    let mut st = RunStats::default();
    let send = |ws: &mut tungstenite::WebSocket<TcpStream>, m: &Message| {
        let data = bincode::serialize(m).unwrap();
        let _ = ws.send(tungstenite::Message::Binary(data));
    };
    while Instant::now() < until {
        let data = match ws.read() {
            Ok(tungstenite::Message::Binary(d)) => d,
            Ok(_) => continue,
            Err(tungstenite::Error::Io(e))
                if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) =>
            {
                continue
            }
            Err(e) => {
                eprintln!("client: connection ended: {e}");
                break;
            }
        };
        let now = Instant::now();
        let measuring = now >= measure_from;
        if measuring {
            st.bytes += data.len() as u64;
        }
        let Ok(msg) = bincode::deserialize::<Message>(&data) else {
            continue;
        };
        match msg {
            Message::Ping => send(&mut ws, &Message::Pong),
            Message::VideoFrame { sequence, frame } => {
                let decoded = decoder.decode(&frame.data);
                let done = Instant::now();
                let marker = match decoded {
                    Ok(Some(yuv)) => {
                        use openh264::formats::YUVSource;
                        let (stride, _, _) = yuv.strides();
                        let m = read_marker(yuv.y(), stride);
                        // Sample quality on every 5th decoded frame.
                        if let Some(ms) = m.filter(|_| measuring && st.frames % 5 == 0) {
                            let ys = yuv.y();
                            let mut compact = Vec::with_capacity(w * h);
                            for row in 0..h {
                                compact.extend_from_slice(&ys[row * stride..row * stride + w]);
                            }
                            let _ = quality.try_send((ms, compact));
                        }
                        m
                    }
                    Ok(None) => None,
                    Err(_) => {
                        if measuring {
                            st.decode_errors += 1;
                        }
                        None
                    }
                };
                if measuring && frame.data.is_empty() {
                    st.skipped += 1;
                } else if measuring {
                    st.frames += 1;
                    if frame.is_keyframe {
                        st.keyframes += 1;
                    }
                    match marker {
                        Some(ms) => {
                            let now_ms = done.duration_since(epoch).as_secs_f64() * 1000.0;
                            let lat = now_ms - ms as f64;
                            st.latencies_ms.push(lat);
                            // WAN_BENCH_TRACE=<ms>: log frames slower than that,
                            // with UTC wall time to line up with server logs.
                            if let Some(t) = trace_ms() {
                                if lat >= t {
                                    eprintln!(
                                        "SPIKE {} seq={sequence} lat={lat:.0}ms bytes={} kf={}",
                                        utc_now(),
                                        frame.data.len(),
                                        frame.is_keyframe
                                    );
                                }
                            }
                            let sec = done.duration_since(measure_from).as_secs();
                            st.per_second_unique.entry(sec).or_default().insert(ms);
                        }
                        None => st.marker_misses += 1,
                    }
                }
                #[cfg(not(phantom_baseline))]
                if mode == Mode::Ack {
                    send(&mut ws, &Message::VideoAck { sequence });
                }
                let _ = (sequence, mode);
            }
            _ => {}
        }
    }
    let _ = ws.close(None);
    st
}

// ── One run ─────────────────────────────────────────────────────────────────

struct NoInput;
impl InputForwarder for NoInput {
    fn forward_input(&self, _event: &InputEvent) -> Result<()> {
        Ok(())
    }
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[i]
}

fn run_once(c: &Config, link: Link, workload: Workload, mode: Mode) -> String {
    let epoch = Instant::now();
    let server_l = TcpListener::bind("127.0.0.1:0").unwrap();
    let server_addr = server_l.local_addr().unwrap();
    let link_l = TcpListener::bind("127.0.0.1:0").unwrap();
    let link_addr = link_l.local_addr().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let cancel = Arc::new(AtomicBool::new(false));

    let (w, h, fps, bitrate, src_fps) = (c.width, c.height, c.fps, c.bitrate, c.src_fps);
    let encoder_name = c.encoder.clone();
    let cancel_s = cancel.clone();
    let (ready_tx, ready_rx) = mpsc::channel();
    let server = thread::spawn(move || {
        let (stream, _) = server_l.accept().expect("server accept");
        let (sender, receiver) = raw_ws_session(stream).expect("ws session");
        let mut capture = SynthCapture::new(w, h, workload, epoch, src_fps);
        let mut encoder = make_encoder(&encoder_name, w, h, fps as f32, bitrate);
        let mut differ = TileDiffer::new();
        let _ = ready_tx.send(());
        let cfg = SessionConfig {
            sender: Box::new(sender),
            receiver: Box::new(receiver),
            frame_interval: Duration::from_secs_f64(1.0 / fps as f64),
            cancel: cancel_s,
            send_file: None,
            video_codec: VideoCodec::H264,
            is_resume: false,
            input_forwarder: Some(Box::new(NoInput)),
            audio_ws_rx: None,
            resolution_change_fn: None,
            paste_fn: Some(Box::new(|_| {})),
        };
        let _ = run_session_cpu(&mut capture, encoder.as_mut(), &mut differ, cfg);
    });
    spawn_link(link_l, server_addr, link, stop.clone());

    let start = Instant::now();
    let measure_from = start + Duration::from_secs_f64(c.warmup);
    let until = measure_from + Duration::from_secs_f64(c.secs);
    let (qtx, qjoin) = spawn_quality_worker(w, h, workload, src_fps);
    let client = thread::spawn(move || {
        run_client(
            link_addr,
            mode,
            epoch,
            measure_from,
            until,
            qtx,
            w as usize,
            h as usize,
        )
    });
    let _ = ready_rx.recv_timeout(Duration::from_secs(10));
    let mut st = client.join().unwrap();
    let psnrs = qjoin.join().unwrap();
    let psnr = if psnrs.is_empty() {
        f64::NAN
    } else {
        psnrs.iter().sum::<f64>() / psnrs.len() as f64
    };
    cancel.store(true, Ordering::Relaxed);
    stop.store(true, Ordering::Relaxed);
    let _ = server.join();

    st.latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let unique: usize = st.per_second_unique.values().map(|s| s.len()).sum();
    let fps_avg = unique as f64 / c.secs;
    let worst = (0..c.secs as u64)
        .map(|s| st.per_second_unique.get(&s).map_or(0, |v| v.len()))
        .min()
        .unwrap_or(0);
    let mbps = st.bytes as f64 * 8.0 / c.secs / 1e6;
    let (p50, p95, p99) = (
        pct(&st.latencies_ms, 0.50),
        pct(&st.latencies_ms, 0.95),
        pct(&st.latencies_ms, 0.99),
    );
    let mode_s = match mode {
        Mode::Ack => "ack",
        Mode::NoAck => "noack",
    };
    let wl = format!("{workload:?}").to_lowercase();
    println!(
        "{:<14} {:<7} {:<6} fps {:>5.1} (worst {:>3})  lat p50 {:>7.0} p95 {:>7.0} p99 {:>7.0} ms  {:>5.2} Mbps  PSNR {:>5.1} dB  kf {:>3}  err {} miss {} skip {}",
        link.name(),
        wl,
        mode_s,
        fps_avg,
        worst,
        p50,
        p95,
        p99,
        mbps,
        psnr,
        st.keyframes,
        st.decode_errors,
        st.marker_misses,
        st.skipped,
    );
    format!(
        "{{\"link\":\"{}\",\"workload\":\"{wl}\",\"mode\":\"{mode_s}\",\"fps\":{fps_avg:.2},\"worst_fps\":{worst},\"p50\":{p50:.1},\"p95\":{p95:.1},\"p99\":{p99:.1},\"mbps\":{mbps:.3},\"psnr\":{psnr:.2},\"keyframes\":{},\"decode_errors\":{},\"marker_misses\":{},\"skipped\":{},\"frames\":{}}}",
        link.name(),
        st.keyframes,
        st.decode_errors,
        st.marker_misses,
        st.skipped,
        st.frames
    )
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .init();
    let c = parse_args();
    println!(
        "phantom wan_bench: {}x{} source {} fps, server {} fps, {} {} kbps start, {}s per run (+{}s warmup), queue {} ms{}",
        c.width,
        c.height,
        c.src_fps,
        c.fps,
        c.encoder,
        c.bitrate,
        c.secs,
        c.warmup,
        c.links.first().map_or(0, |l| l.queue_ms),
        if cfg!(phantom_baseline) { ", BASELINE build" } else { "" }
    );
    for _ in 0..c.reps {
        for &link in &c.links {
            for &workload in &c.workloads {
                for &mode in &c.modes {
                    let json = run_once(&c, link, workload, mode);
                    println!("JSON {json}");
                }
            }
        }
    }
}

fn make_encoder(
    name: &str,
    w: u32,
    h: u32,
    fps: f32,
    kbps: u32,
) -> Box<dyn phantom_core::encode::FrameEncoder> {
    match name {
        "openh264" => Box::new(OpenH264Encoder::new(w, h, fps, kbps).expect("openh264")),
        #[cfg(feature = "x264")]
        "x264" => phantom_server::encode::x264::boxed(w, h, fps, kbps).expect("x264"),
        other => panic!("encoder {other} not available in this build (x264 needs --features x264)"),
    }
}
