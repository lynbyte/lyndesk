//! Capture benchmark: how long one screen grab takes, per backend.
//!
//! ```text
//! cargo run --release -p phantom-server --example capture_bench -- [--cuda] [--secs 5] [scrap|gdi ...]
//! ```
//!
//! `--cuda` initialises CUDA first, as the server's GPU probe does — on
//! hybrid-GPU (Optimus) laptops that ties the process to the discrete GPU,
//! and Desktop Duplication of a display driven by the integrated GPU then
//! fails. Something on screen should be moving (a video) so DXGI-based
//! capture has new frames to return; GDI copies the screen every call.

use phantom_core::capture::FrameCapture;
use std::time::{Duration, Instant};

fn open(name: &str) -> anyhow::Result<Box<dyn FrameCapture>> {
    Ok(match name {
        "scrap" => Box::new(phantom_server::capture::scrap::ScrapCapture::new()?),
        #[cfg(windows)]
        "gdi" => Box::new(phantom_server::capture::gdi::GdiCapture::new_primary_monitor()?),
        other => anyhow::bail!("unknown capture {other}"),
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cuda = args.iter().any(|a| a == "--cuda");
    let secs = args
        .iter()
        .position(|a| a == "--secs")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(5.0);
    let mut names: Vec<&str> = args
        .iter()
        .map(|s| s.as_str())
        .filter(|a| matches!(*a, "scrap" | "gdi"))
        .collect();
    if names.is_empty() {
        names = vec!["scrap", "gdi"];
    }
    if cuda {
        match phantom_gpu::cuda::CudaLib::load() {
            Ok(_lib) => println!("CUDA initialised (as the server's GPU probe does)"),
            Err(e) => println!("CUDA not available: {e}"),
        }
    }
    for name in names {
        let mut cap = match open(name) {
            Ok(c) => c,
            Err(e) => {
                println!("{name:<6} open failed: {e:#}");
                continue;
            }
        };
        let (w, h) = cap.resolution();
        let start = Instant::now();
        let (mut frames, mut empty, mut errors) = (0u32, 0u32, 0u32);
        let mut grab_ms = Vec::new();
        let mut last_err = None;
        while start.elapsed() < Duration::from_secs_f64(secs) {
            let t = Instant::now();
            match cap.capture() {
                Ok(Some(_)) => {
                    frames += 1;
                    grab_ms.push(t.elapsed().as_secs_f64() * 1000.0);
                }
                Ok(None) => {
                    empty += 1;
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(e) => {
                    errors += 1;
                    last_err = Some(format!("{e:#}"));
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        }
        grab_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p = |q: f64| {
            grab_ms
                .get(((grab_ms.len() as f64 - 1.0) * q) as usize)
                .copied()
                .unwrap_or(f64::NAN)
        };
        println!(
            "{name:<6} {w}x{h}: {:.1} frames/s, grab p50 {:.1} ms p95 {:.1} ms (max {:.0} fps), {empty} no-change polls, {errors} errors{}",
            frames as f64 / secs,
            p(0.5),
            p(0.95),
            1000.0 / p(0.5),
            last_err.map(|e| format!(" (last: {e})")).unwrap_or_default()
        );
    }
}
