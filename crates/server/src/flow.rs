//! Ack-based video flow control.
//!
//! The client acknowledges each video frame once it has been decoded and
//! painted (`Message::VideoAck`). The server keeps a small window of frames
//! "in flight" — enough to cover one round trip plus decode — and stops
//! producing new frames while the window is full. Because the capture is not
//! ticked while blocked, the next frame naturally carries every change made
//! in the meantime: under congestion the stream drops to the rate the path can
//! actually deliver instead of piling frames up in socket buffers.
//!
//! Why this is needed: on WSS the sender only pushes into a bounded channel,
//! so the time a `send_msg` takes says nothing about the network. Latency
//! builds up in kernel and proxy buffers long before that channel fills.
//!
//! Flow control stays off until the first ack arrives, so transports and
//! clients that never ack (WebRTC media tracks, native clients, older web
//! bundles) keep their previous behaviour.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// A client whose oldest unacknowledged frame is older than this is treated
/// as stalled (hidden tab, frozen network): its window is cleared and it is
/// resynchronised with a keyframe instead of gating the stream forever.
pub const STALL_AFTER: Duration = Duration::from_secs(4);

/// How long the minimum send→ack delay is trusted before it may rise again.
const BASE_DELAY_WINDOW: Duration = Duration::from_secs(10);

/// Interval over which the minimum delay is taken as the standing queue
/// (CoDel's idea: transient bursts don't raise the minimum, a queue does).
const QUEUE_WINDOW: Duration = Duration::from_millis(200);

/// Window of the max-delivery-rate filter (capacity estimate).
const MAX_RATE_WINDOW: Duration = Duration::from_secs(10);

/// Smallest in-flight byte budget, so ordinary frames are never held back
/// by a capacity estimate taken while the screen was nearly idle.
const MIN_BYTE_BUDGET: usize = 48 * 1024;

/// In-flight byte budget in round trips of capacity (BBR's cwnd gain).
const BYTE_BUDGET_GAIN: f64 = 2.0;

/// Delay assumed before any ack has been measured.
const DEFAULT_RTT_MS: f64 = 40.0;

struct Inflight {
    sequence: u64,
    sent: Instant,
    bytes: usize,
}

/// What the session loop should do with the next frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowVerdict {
    /// Window open (or flow control inactive): produce a frame.
    Send,
    /// Window full: do not capture/encode yet.
    Wait,
    /// The client stopped acking; the window was reset and the caller must
    /// force a keyframe so the client can resynchronise.
    Stalled,
}

pub struct FlowControl {
    active: bool,
    inflight: VecDeque<Inflight>,
    /// Minimum send→ack delay (ms): path RTT plus decode, without queueing.
    base_ms: f64,
    base_at: Instant,
    /// Standing queue (ms): minimum delay over `QUEUE_WINDOW` above `base_ms`.
    queue_ms: f64,
    /// Recent (ack time, send→ack ms) samples for the windowed minimum.
    recent: VecDeque<(Instant, f64)>,
    /// Delivery rate measured from acked bytes (bits/s, EWMA over 0.5s windows).
    rate_bps: f64,
    /// Highest 0.5 s delivery rate within `MAX_RATE_WINDOW`: the capacity
    /// estimate behind the byte budget.
    max_rate_bps: f64,
    max_rate_at: Instant,
    rate_window_start: Instant,
    rate_window_bytes: usize,
    /// Counters for the stats line.
    pub blocked_ticks: u64,
    pub stalls: u64,
}

impl Default for FlowControl {
    fn default() -> Self {
        Self::new()
    }
}

impl FlowControl {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            active: false,
            inflight: VecDeque::new(),
            base_ms: f64::MAX,
            base_at: now,
            queue_ms: 0.0,
            recent: VecDeque::new(),
            rate_bps: 0.0,
            max_rate_bps: 0.0,
            max_rate_at: now,
            rate_window_start: now,
            rate_window_bytes: 0,
            blocked_ticks: 0,
            stalls: 0,
        }
    }

    /// True once the client has acked at least one frame.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Record a frame handed to the transport.
    pub fn on_sent(&mut self, sequence: u64, bytes: usize) {
        self.inflight.push_back(Inflight {
            sequence,
            sent: Instant::now(),
            bytes,
        });
        // Without acks nothing ever drains the queue; keep it bounded.
        if !self.active && self.inflight.len() > 256 {
            self.inflight.pop_front();
        }
    }

    /// A cumulative ack: every frame up to and including `sequence` is done.
    pub fn on_ack(&mut self, sequence: u64) {
        self.on_ack_at(sequence, Instant::now());
    }

    fn on_ack_at(&mut self, sequence: u64, now: Instant) {
        self.active = true;
        let mut acked = 0usize;
        let mut sample = None;
        while let Some(f) = self.inflight.front() {
            if f.sequence > sequence {
                break;
            }
            acked += f.bytes;
            if f.sequence == sequence {
                sample = Some(now.duration_since(f.sent).as_secs_f64() * 1000.0);
            }
            self.inflight.pop_front();
        }
        if let Some(ms) = sample {
            // Standing queue, measured CoDel-style: the MINIMUM delay over the
            // last QUEUE_WINDOW, minus the base. One big frame (a 20 KB frame
            // is 32 ms on 5 Mbps however empty the path) raises only its own
            // sample; a small frame next to it still gets through fast. Only a
            // real queue delays every frame. Two earlier attempts subtracted
            // per-frame transmission time from a rate estimate instead, and
            // both broke while the sender is app-limited (throughput, even its
            // windowed max, understates the link): base fell below the RTT
            // and every frame read as 40 ms queued.
            self.recent.push_back((now, ms));
            while self
                .recent
                .front()
                .is_some_and(|(t, _)| now.duration_since(*t) > QUEUE_WINDOW)
            {
                self.recent.pop_front();
            }
            let recent_min = self.recent.iter().map(|(_, d)| *d).fold(f64::MAX, f64::min);
            if recent_min < self.base_ms || now.duration_since(self.base_at) >= BASE_DELAY_WINDOW {
                self.base_ms = recent_min;
                self.base_at = now;
            }
            self.queue_ms = (recent_min - self.base_ms).max(0.0);
        }
        self.rate_window_bytes += acked;
        let elapsed = now.duration_since(self.rate_window_start).as_secs_f64();
        if elapsed >= 0.5 {
            let r = self.rate_window_bytes as f64 * 8.0 / elapsed;
            self.rate_bps = if self.rate_bps == 0.0 {
                r
            } else {
                self.rate_bps * 0.6 + r * 0.4
            };
            if r >= self.max_rate_bps || now.duration_since(self.max_rate_at) >= MAX_RATE_WINDOW {
                self.max_rate_bps = r;
                self.max_rate_at = now;
            }
            self.rate_window_bytes = 0;
            self.rate_window_start = now;
        }
    }

    /// Forget everything in flight (stream restarted, resume, stall).
    pub fn reset_window(&mut self) {
        self.inflight.clear();
    }

    /// Unloaded round trip (ms) as seen through acks, or a default.
    pub fn base_rtt_ms(&self) -> f64 {
        if self.base_ms.is_finite() && self.base_ms < f64::MAX {
            self.base_ms
        } else {
            DEFAULT_RTT_MS
        }
    }

    /// Smoothed queueing delay (ms).
    pub fn queue_ms(&self) -> f64 {
        self.queue_ms
    }

    /// Measured delivery rate (kbit/s); 0 until measured.
    pub fn delivered_kbps(&self) -> f64 {
        self.rate_bps / 1000.0
    }

    /// Frames in flight and age of the oldest one.
    pub fn backlog(&self) -> (usize, Duration) {
        (
            self.inflight.len(),
            self.inflight
                .front()
                .map(|f| f.sent.elapsed())
                .unwrap_or_default(),
        )
    }

    /// Decide whether the next frame may be produced.
    pub fn check(&mut self, frame_interval: Duration) -> FlowVerdict {
        self.check_at(frame_interval, Instant::now())
    }

    fn check_at(&mut self, frame_interval: Duration, now: Instant) -> FlowVerdict {
        if !self.active {
            return FlowVerdict::Send;
        }
        let Some(front) = self.inflight.front() else {
            return FlowVerdict::Send;
        };
        let age = now.duration_since(front.sent);
        if age >= STALL_AFTER {
            self.inflight.clear();
            self.stalls += 1;
            return FlowVerdict::Stalled;
        }
        let rtt = self.base_rtt_ms();
        let interval_ms = frame_interval.as_secs_f64().max(0.001) * 1000.0;
        // Enough frames to cover one round trip plus decode, never more than
        // ~rtt+120ms of queue (same budget voxl uses).
        let max_frames = (((rtt + 40.0) / interval_ms).ceil() as usize + 2).clamp(3, 40);
        let max_age = Duration::from_secs_f64((rtt + 120.0) / 1000.0);
        // Byte budget: BYTE_BUDGET_GAIN round trips of the path's capacity.
        // A frame count alone lets four more frames pile up behind one 400 KB
        // full-screen change (each then arrives 400+ ms late); with the
        // budget the loop waits for the big frame and the next one it sends
        // shows the latest screen. One BDP is too tight — the pipe never
        // fills (2 Mbps scroll fell to 7 fps) — so, like BBR's cwnd gain,
        // allow two. The floor keeps it from throttling small frames while
        // the capacity estimate is still app-limited.
        let inflight_bytes: usize = self.inflight.iter().map(|f| f.bytes).sum();
        let max_bytes = if self.max_rate_bps > 0.0 {
            ((self.max_rate_bps / 8.0) * BYTE_BUDGET_GAIN * (rtt + 40.0) / 1000.0)
                .max(MIN_BYTE_BUDGET as f64) as usize
        } else {
            usize::MAX
        };
        if self.inflight.len() >= max_frames || age > max_age || inflight_bytes >= max_bytes {
            self.blocked_ticks += 1;
            FlowVerdict::Wait
        } else {
            FlowVerdict::Send
        }
    }
}

// ── Delay-based rate control ────────────────────────────────────────────────

/// Delay-based bitrate adaptation driven by the ack measurements above.
///
/// Backs off toward the measured delivery rate as soon as acks show frames
/// queueing on the path, and probes upward while the path is clear and the
/// content actually presses against the cap. Evaluated every 100 ms — the
/// ping-based `AdaptiveBitrate` only reacts every 5 s and cannot tell a slow
/// link from a busy one. Used instead of it once the client acks frames.
pub struct RateCtl {
    target: f64,
    max: f64,
    min: f64,
    last_eval: Instant,
    last_dec: Instant,
    last_inc: Instant,
    /// Ramp multiplicatively until the first sign of congestion (like TCP
    /// slow start), so a session neither floods a slow link nor crawls up on
    /// a fast one.
    slow_start: bool,
    /// Delivery rate seen at the last congestion: the path's likely capacity.
    ceiling: f64,
    ceiling_at: Instant,
    applied: u32,
}

impl RateCtl {
    /// `initial_kbps` is the encoder's configured bitrate; bounds follow the
    /// ping-based controller's (min clamp 500..1500, max 3x initial).
    pub fn for_initial(initial_kbps: u32) -> Self {
        let initial = initial_kbps.max(1) as f64;
        Self::new(
            initial,
            initial_kbps.clamp(500, 1500) as f64,
            (initial * 3.0).max(initial),
        )
    }

    pub fn new(start: f64, min: f64, max: f64) -> Self {
        let past = Instant::now() - Duration::from_secs(10);
        let target = start.clamp(min, max);
        Self {
            target,
            max,
            min,
            last_eval: past,
            last_dec: past,
            last_inc: past,
            slow_start: true,
            ceiling: f64::MAX,
            ceiling_at: past,
            applied: target as u32,
        }
    }

    pub fn target_kbps(&self) -> u32 {
        self.target as u32
    }

    /// Re-evaluate; returns a new encoder bitrate when it moved by ≥2%.
    pub fn update(&mut self, flow: &FlowControl) -> Option<u32> {
        let (_, oldest) = flow.backlog();
        self.update_with(
            Instant::now(),
            flow.queue_ms(),
            oldest.as_secs_f64() * 1000.0,
            flow.base_rtt_ms(),
            flow.delivered_kbps(),
        )
    }

    fn update_with(
        &mut self,
        now: Instant,
        queue_ms: f64,
        oldest_ms: f64,
        rtt_ms: f64,
        delivered: f64,
    ) -> Option<u32> {
        if now.duration_since(self.last_eval) < Duration::from_millis(100) {
            return None;
        }
        self.last_eval = now;
        let congested = queue_ms > 30.0 || oldest_ms > rtt_ms + 80.0;
        let clear = queue_ms <= 15.0;
        if congested && now.duration_since(self.last_dec) >= Duration::from_millis(300) {
            let mut t = self.target * 0.85;
            // Delivery only says something about capacity when we were
            // pushing near the target; one big frame on an idle screen would
            // otherwise pin the ceiling at a few kbps.
            if delivered > 0.0 && delivered >= self.target * 0.5 {
                self.ceiling = delivered;
                self.ceiling_at = now;
                t = t.min(delivered * 0.85);
            }
            self.target = t.max(self.min);
            self.last_dec = now;
            self.slow_start = false;
        } else if self.slow_start
            && !congested
            && clear
            && now.duration_since(self.last_inc) >= Duration::from_millis(250)
        {
            // Only probe upward once the content presses against the cap.
            if delivered >= self.target * 0.5 {
                self.target = (self.target * 1.5).min(self.max);
                self.last_inc = now;
            }
        } else if !congested
            && clear
            && now.duration_since(self.last_dec) >= Duration::from_secs(1)
            && delivered >= self.target * 0.5
        {
            // Paths change: forget a capacity estimate not confirmed lately.
            if now.duration_since(self.ceiling_at) >= Duration::from_secs(15) {
                self.ceiling = f64::MAX;
            }
            // Recover quickly to just under the known capacity, then probe
            // past it gently so a busy link isn't pushed into its queue
            // every couple of seconds.
            if self.target < self.ceiling * 0.9
                && now.duration_since(self.last_inc) >= Duration::from_millis(200)
            {
                self.target = (self.target * 1.06 + 100.0)
                    .min(self.ceiling * 0.9)
                    .min(self.max);
                self.last_inc = now;
            } else if self.target >= self.ceiling * 0.9
                && now.duration_since(self.last_inc) >= Duration::from_millis(1000)
            {
                self.target = (self.target * 1.02).min(self.max);
                self.last_inc = now;
            }
        }
        self.target = self.target.clamp(self.min, self.max);
        let t = self.target as u32;
        let moved = (t as f64 - self.applied as f64).abs() / self.applied.max(1) as f64;
        if moved >= 0.02 {
            self.applied = t;
            Some(t)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: Duration = Duration::from_millis(16);

    #[test]
    fn rate_backs_off_toward_delivery_on_queueing() {
        let mut r = RateCtl::new(8000.0, 1000.0, 30000.0);
        let t0 = Instant::now();
        // Queue building while ~5 Mbps gets through: drop under delivery.
        let k = r.update_with(t0, 80.0, 200.0, 40.0, 5000.0).unwrap();
        assert!(k <= 4250, "expected to drop under delivery, got {k}");
        // Repeated congestion inside 300ms does not stack decreases.
        assert_eq!(
            r.update_with(t0 + Duration::from_millis(150), 80.0, 200.0, 40.0, 5000.0),
            None
        );
        // Delivery far below target says little about capacity (one big
        // frame on an idle screen): plain multiplicative decrease.
        let mut r = RateCtl::new(8000.0, 1000.0, 30000.0);
        assert_eq!(r.update_with(t0, 80.0, 200.0, 40.0, 1000.0), Some(6800));
    }

    #[test]
    fn rate_slow_starts_only_when_content_uses_the_cap() {
        let mut r = RateCtl::new(4000.0, 1000.0, 30000.0);
        let t0 = Instant::now();
        // Idle screen: delivering little, no probe.
        assert_eq!(r.update_with(t0, 0.0, 0.0, 20.0, 100.0), None);
        // Busy screen on a clear path: ramp ×1.5.
        let k = r
            .update_with(t0 + Duration::from_millis(300), 0.0, 0.0, 20.0, 3500.0)
            .unwrap();
        assert_eq!(k, 6000);
    }

    #[test]
    fn rate_recovers_below_known_ceiling_then_probes_gently() {
        let mut r = RateCtl::new(10000.0, 1000.0, 30000.0);
        let mut t = Instant::now();
        r.update_with(t, 60.0, 100.0, 20.0, 9000.0); // ceiling 9000, target 7650
        let mut last = r.target_kbps();
        for _ in 0..40 {
            t += Duration::from_millis(250);
            r.update_with(t + Duration::from_secs(1), 0.0, 0.0, 20.0, 9000.0);
            assert!(r.target_kbps() >= last);
            last = r.target_kbps();
        }
        // Climbs back to 0.9 x ceiling fast, then probes past it at only
        // ~2%/s: well under 1.1 x ceiling after ten seconds.
        assert!((8100..=9900).contains(&last), "target {last}");
    }

    #[test]
    fn rate_respects_bounds() {
        let mut r = RateCtl::new(2000.0, 1500.0, 3000.0);
        let mut t = Instant::now();
        for _ in 0..20 {
            t += Duration::from_millis(400);
            r.update_with(t, 200.0, 500.0, 20.0, 100.0);
        }
        assert_eq!(r.target_kbps(), 1500);
        let mut r = RateCtl::new(2000.0, 1500.0, 3000.0);
        for _ in 0..20 {
            t += Duration::from_millis(400);
            r.update_with(t, 0.0, 0.0, 20.0, 5000.0);
        }
        assert_eq!(r.target_kbps(), 3000);
    }

    #[test]
    fn inactive_until_first_ack() {
        let mut f = FlowControl::new();
        for seq in 1..=100 {
            f.on_sent(seq, 1000);
        }
        assert!(!f.is_active());
        assert_eq!(f.check(FRAME), FlowVerdict::Send);
    }

    #[test]
    fn window_full_blocks_then_ack_reopens() {
        let mut f = FlowControl::new();
        f.on_sent(1, 1000);
        f.on_ack(1);
        assert!(f.is_active());
        // Default 40ms RTT at 16ms frames: ceil(80/16)+2 = 7 frames.
        for seq in 2..=8 {
            f.on_sent(seq, 1000);
        }
        assert_eq!(f.check(FRAME), FlowVerdict::Wait);
        f.on_ack(5);
        assert_eq!(f.backlog().0, 3);
        assert_eq!(f.check(FRAME), FlowVerdict::Send);
    }

    #[test]
    fn ack_is_cumulative_and_ignores_unknown() {
        let mut f = FlowControl::new();
        for seq in 1..=5 {
            f.on_sent(seq, 100);
        }
        f.on_ack(3);
        assert_eq!(f.backlog().0, 2);
        // Ack for an already-acked frame is harmless.
        f.on_ack(2);
        assert_eq!(f.backlog().0, 2);
        f.on_ack(10);
        assert_eq!(f.backlog().0, 0);
    }

    #[test]
    fn old_frame_blocks_on_age() {
        let mut f = FlowControl::new();
        f.on_sent(1, 100);
        f.on_ack(1);
        f.on_sent(2, 100);
        let later = Instant::now() + Duration::from_millis(400);
        assert_eq!(f.check_at(FRAME, later), FlowVerdict::Wait);
    }

    #[test]
    fn stalled_client_resets_window() {
        let mut f = FlowControl::new();
        f.on_sent(1, 100);
        f.on_ack(1);
        f.on_sent(2, 100);
        f.on_sent(3, 100);
        let later = Instant::now() + STALL_AFTER + Duration::from_millis(1);
        assert_eq!(f.check_at(FRAME, later), FlowVerdict::Stalled);
        assert_eq!(f.backlog().0, 0);
        assert_eq!(f.stalls, 1);
        assert_eq!(f.check(FRAME), FlowVerdict::Send);
    }

    /// Ack `seq`, sent at `sent`, after `delay_ms`.
    fn ack_after(f: &mut FlowControl, seq: u64, sent: Instant, delay_ms: u64) -> Instant {
        f.inflight.push_back(Inflight {
            sequence: seq,
            sent,
            bytes: 100,
        });
        let at = sent + Duration::from_millis(delay_ms);
        f.on_ack_at(seq, at);
        at
    }

    #[test]
    fn byte_budget_holds_frames_behind_a_huge_one() {
        let mut f = FlowControl::new();
        f.on_sent(1, 100);
        f.on_ack(1);
        // Capacity estimate: 10 Mbps.
        f.max_rate_bps = 10e6;
        // One 400 KB full-screen change in flight: far over one RTT of link.
        f.on_sent(2, 400 * 1024);
        assert_eq!(f.check(FRAME), FlowVerdict::Wait);
        f.on_ack(2);
        assert_eq!(f.check(FRAME), FlowVerdict::Send);
        // Ordinary frames stay under the 48 KB floor even at a tiny estimate.
        f.max_rate_bps = 100e3;
        f.on_sent(3, 8 * 1024);
        f.on_sent(4, 8 * 1024);
        assert_eq!(f.check(FRAME), FlowVerdict::Send);
    }

    #[test]
    fn one_big_frame_is_not_a_queue() {
        let mut f = FlowControl::new();
        let t0 = Instant::now();
        ack_after(&mut f, 1, t0, 20);
        assert!((f.base_rtt_ms() - 20.0).abs() < 0.5);
        // A keyframe takes 100ms longer, but the next small frame is fast.
        ack_after(&mut f, 2, t0 + Duration::from_millis(33), 120);
        ack_after(&mut f, 3, t0 + Duration::from_millis(66), 22);
        assert!(f.queue_ms() < 5.0, "queue {}", f.queue_ms());
    }

    #[test]
    fn standing_queue_raises_the_windowed_minimum() {
        let mut f = FlowControl::new();
        let t0 = Instant::now();
        ack_after(&mut f, 1, t0, 20);
        // Every frame for >200ms waits 80ms extra: a real queue.
        for i in 0..12u64 {
            ack_after(&mut f, 2 + i, t0 + Duration::from_millis(50 + i * 33), 100);
        }
        assert!((f.queue_ms() - 80.0).abs() < 1.0, "queue {}", f.queue_ms());
        assert!((f.base_rtt_ms() - 20.0).abs() < 0.5);
    }
}
