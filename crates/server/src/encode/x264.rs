//! x264 encoder (optional `x264` feature; GPL — see scripts/build-x264.sh).
//!
//! Tuned like voxl's: CRF quality with a VBV cap of 1.25 frames, so every
//! frame stays near one frame's budget — a full-screen change is coded at a
//! higher QP instead of as a ~400 KB frame (OpenH264's rate control can only
//! move QP a few steps per frame, which is what left phantom's scroll p99
//! above the old path at 10–20 Mbps; docs/benchmarks/2026-11-09-wan.md).
//! UMH motion search follows scrolls and window drags; no scene-cut
//! keyframes; zerolatency, no B-frames, constrained baseline for OpenH264
//! and WebCodecs decoders. `set_bitrate_kbps` moves the VBV cap in place.
//!
//! All x264 struct handling lives in `csrc/x264_shim.c`.

use anyhow::{bail, Context, Result};
use phantom_core::encode::{EncodedFrame, FrameEncoder, VideoCodec};
use phantom_core::frame::Frame;
use std::ffi::{c_char, c_float, c_int, CString};

#[repr(C)]
struct PhX264 {
    _private: [u8; 0],
}

extern "C" {
    fn phx264_open(
        width: c_int,
        height: c_int,
        fps: c_int,
        threads: c_int,
        max_kbps: c_int,
        crf: c_float,
        preset: *const c_char,
    ) -> *mut PhX264;
    #[allow(clippy::too_many_arguments)]
    fn phx264_encode(
        e: *mut PhX264,
        y: *const u8,
        u: *const u8,
        v: *const u8,
        stride_y: c_int,
        stride_uv: c_int,
        force_idr: c_int,
        out: *mut *mut u8,
        keyframe: *mut c_int,
    ) -> c_int;
    fn phx264_set_max_kbps(e: *mut PhX264, max_kbps: c_int) -> c_int;
    fn phx264_close(e: *mut PhX264);
}

/// Base quality (x264 CRF). voxl ships 24; lower is sharper.
const CRF: f32 = 23.0;

pub struct X264Encoder {
    raw: *mut PhX264,
    width: u32,
    height: u32,
    bitrate_kbps: u32,
    force_idr: bool,
}

// SAFETY: the encoder handle is owned exclusively by this struct and x264
// handles are usable from any (one) thread at a time.
unsafe impl Send for X264Encoder {}

impl X264Encoder {
    /// `bitrate_kbps` is the VBV cap; quality comes from CRF below it.
    pub fn new(width: u32, height: u32, fps: f32, bitrate_kbps: u32) -> Result<Self> {
        if width % 2 != 0 || height % 2 != 0 {
            bail!("x264 needs even dimensions for I420, got {width}x{height}");
        }
        let ncpu = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        // Sliced threads cut latency, but their overhead grows with count.
        let threads = (ncpu / 2).clamp(1, 4) as c_int;
        let preset = CString::new("ultrafast").expect("static preset");
        // SAFETY: plain arguments; the shim copies `preset` before returning.
        let raw = unsafe {
            phx264_open(
                width as c_int,
                height as c_int,
                fps.round().max(1.0) as c_int,
                threads,
                bitrate_kbps as c_int,
                CRF,
                preset.as_ptr(),
            )
        };
        if raw.is_null() {
            bail!("x264_encoder_open failed ({width}x{height} @ {fps} fps)");
        }
        tracing::info!(
            width,
            height,
            fps,
            bitrate_kbps,
            threads,
            crf = CRF,
            "x264 encoder initialized (GPL build)"
        );
        Ok(Self {
            raw,
            width,
            height,
            bitrate_kbps,
            force_idr: true,
        })
    }
}

impl FrameEncoder for X264Encoder {
    fn encode_frame(&mut self, frame: &Frame) -> Result<EncodedFrame> {
        let (w, h) = (self.width as usize, self.height as usize);
        if frame.width != self.width || frame.height != self.height {
            bail!(
                "x264: frame {}x{} does not match encoder {}x{}",
                frame.width,
                frame.height,
                self.width,
                self.height
            );
        }
        let (y, u, v) = phantom_core::color::bgra_to_yuv420(&frame.data, w, h);
        let mut out: *mut u8 = std::ptr::null_mut();
        let mut keyframe: c_int = 0;
        // SAFETY: the planes outlive the call; strides match their layout
        // (w for Y, w/2 for U and V); `out` points into x264's buffer, which
        // stays valid until the next encode and is copied below.
        let size = unsafe {
            phx264_encode(
                self.raw,
                y.as_ptr(),
                u.as_ptr(),
                v.as_ptr(),
                w as c_int,
                (w / 2) as c_int,
                self.force_idr as c_int,
                &mut out,
                &mut keyframe,
            )
        };
        if size < 0 {
            bail!("x264_encoder_encode failed ({size})");
        }
        self.force_idr = false;
        let data = if size > 0 && !out.is_null() {
            // SAFETY: x264 returned `size` contiguous bytes at `out`.
            unsafe { std::slice::from_raw_parts(out, size as usize) }.to_vec()
        } else {
            Vec::new()
        };
        Ok(EncodedFrame {
            codec: VideoCodec::H264,
            data,
            is_keyframe: keyframe != 0,
        })
    }

    fn force_keyframe(&mut self) {
        self.force_idr = true;
    }

    fn set_bitrate_kbps(&mut self, kbps: u32) -> Result<()> {
        if kbps == self.bitrate_kbps {
            return Ok(());
        }
        // SAFETY: valid handle; plain argument.
        let rc = unsafe { phx264_set_max_kbps(self.raw, kbps as c_int) };
        if rc < 0 {
            bail!("x264_encoder_reconfig failed ({rc})");
        }
        self.bitrate_kbps = kbps;
        Ok(())
    }

    fn bitrate_kbps(&self) -> u32 {
        self.bitrate_kbps
    }
}

impl Drop for X264Encoder {
    fn drop(&mut self) {
        // SAFETY: handle from phx264_open, closed exactly once.
        unsafe { phx264_close(self.raw) };
    }
}

/// Construct by name for callers that already hold a `Box<dyn FrameEncoder>`.
pub fn boxed(
    width: u32,
    height: u32,
    fps: f32,
    bitrate_kbps: u32,
) -> Result<Box<dyn FrameEncoder>> {
    X264Encoder::new(width, height, fps, bitrate_kbps)
        .map(|e| Box::new(e) as Box<dyn FrameEncoder>)
        .context("x264 encoder")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: u32, h: u32, i: u32) -> Frame {
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let a = (x + i * 5) ^ (y + i * 3);
                data.extend_from_slice(&[a as u8, (x * 3 + y) as u8, (y * 2) as u8, 255]);
            }
        }
        Frame {
            width: w,
            height: h,
            format: phantom_core::frame::PixelFormat::Bgra8,
            data,
            timestamp: std::time::Instant::now(),
        }
    }

    #[test]
    fn x264_first_frame_is_idr_then_only_on_request() {
        let mut enc = X264Encoder::new(320, 240, 30.0, 3000).unwrap();
        let first = enc.encode_frame(&frame(320, 240, 0)).unwrap();
        assert!(first.is_keyframe && !first.data.is_empty());
        for i in 1..30 {
            assert!(!enc.encode_frame(&frame(320, 240, i)).unwrap().is_keyframe);
        }
        enc.force_keyframe();
        assert!(enc.encode_frame(&frame(320, 240, 30)).unwrap().is_keyframe);
        enc.set_bitrate_kbps(800).unwrap();
        assert!(!enc.encode_frame(&frame(320, 240, 31)).unwrap().is_keyframe);
    }

    #[test]
    fn x264_output_decodes_with_openh264() {
        let mut enc = X264Encoder::new(320, 240, 30.0, 3000).unwrap();
        let mut dec = openh264::decoder::Decoder::new().unwrap();
        let mut decoded = 0;
        for i in 0..10 {
            let f = enc.encode_frame(&frame(320, 240, i)).unwrap();
            if let Ok(Some(_)) = dec.decode(&f.data) {
                decoded += 1;
            }
        }
        assert!(decoded >= 9, "OpenH264 decoded {decoded}/10 x264 frames");
    }
}
