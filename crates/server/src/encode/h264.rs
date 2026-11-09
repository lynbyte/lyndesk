use anyhow::{Context, Result};
use openh264::encoder::{Encoder, EncoderConfig, FrameType};
use openh264::formats::YUVBuffer;
use phantom_core::encode::{EncodedFrame, FrameEncoder, VideoCodec};
use phantom_core::frame::Frame;

pub struct OpenH264Encoder {
    encoder: Encoder,
    width: u32,
    height: u32,
    fps: f32,
    bitrate_kbps: u32,
    /// Cached Annex-B SPS/PPS prefix from a known-good keyframe.
    ///
    /// WebRTC clients can join right as Windows switches desktops, so every
    /// forced keyframe must be independently decodable.
    sps_pps: Vec<u8>,
    /// The openh264 crate initialises the native encoder lazily on the first
    /// encode; SetOption only works after that.
    encoded_any: bool,
}

/// Lowest frame QP OpenH264 may pick (see `tune_after_init`).
const MIN_QP: i32 = 24;

/// Encoder settings, chosen with `examples/wan_bench.rs` (1080p, 5 Mbps /
/// 60 ms, scroll + full-screen animation):
///
/// - Frame skip OFF. With it on, OpenH264 meets a lower target by dropping
///   whole frames (5 fps while scrolling at 2 Mbps) instead of raising QP.
///   Pacing to the link is the flow-control window's job.
/// - Camera usage, not ScreenContentRealTime: screen mode undershoots its
///   target (2 Mbps on LAN) and scrolled text came out 33 dB vs 45 dB.
/// - Scene-change detection OFF (see `tune_after_init`).
fn encoder_config(fps: f32, bitrate_kbps: u32) -> EncoderConfig {
    EncoderConfig::new()
        .max_frame_rate(fps)
        .set_bitrate_bps(bitrate_kbps * 1000)
        .rate_control_mode(openh264::encoder::RateControlMode::Bitrate)
        .enable_skip_frame(false)
        .usage_type(openh264::encoder::UsageType::CameraVideoRealTime)
}

impl OpenH264Encoder {
    /// Set what the openh264 crate does not expose: scene-change detection
    /// off (scrolling read as a "scene change" and emitted an unrequested
    /// I-frame every ~0.6 s, 13 in 8 s) and a QP floor (`MIN_QP`). Needs the
    /// native encoder to exist, i.e. after the first encode.
    fn tune_after_init(&mut self) {
        let mut p = openh264_sys2::SEncParamExt::default();
        // SAFETY: the encoder is initialised; ENCODER_OPTION_SVC_ENCODE_PARAM_EXT
        // reads/writes an SEncParamExt that outlives both calls.
        unsafe {
            let api = self.encoder.raw_api();
            if api.get_option(
                openh264_sys2::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                std::ptr::addr_of_mut!(p).cast(),
            ) != 0
            {
                tracing::warn!("OpenH264 GetOption(PARAM_EXT) failed");
                return;
            }
            p.bEnableSceneChangeDetect = false;
            // QP floor. OpenH264 moves frame QP only ~3 steps per frame
            // (DELTA_QP_BGD_THD), so after cheap content (scrolling) has let
            // QP sink near 0, a full-screen change comes out as a ~400 KB
            // frame that alone takes 300 ms at 10 Mbps. A floor of 24 (≈
            // voxl's CRF) keeps that frame several times smaller. wan_bench,
            // 3 runs, scroll p99: 20 Mbps 216 → 104–113 ms, 10 Mbps
            // 176 → 111–140 ms (0.5.10: 107–122 / 156–176) at 29 fps; LAN
            // scroll PSNR ~45 → ~38 dB (0.5.10: 30 dB).
            p.iMinQp = MIN_QP;
            let rc = api.set_option(
                openh264_sys2::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                std::ptr::addr_of_mut!(p).cast(),
            );
            if rc != 0 {
                tracing::warn!(rc, "OpenH264: could not disable scene-change detection");
            }
        }
    }

    pub fn new(width: u32, height: u32, fps: f32, bitrate_kbps: u32) -> Result<Self> {
        let config = encoder_config(fps, bitrate_kbps);

        let api = openh264::OpenH264API::from_source();
        let encoder =
            Encoder::with_api_config(api, config).context("failed to create OpenH264 encoder")?;

        tracing::info!(
            width,
            height,
            fps,
            bitrate_kbps,
            "OpenH264 encoder initialized"
        );
        Ok(Self {
            encoder,
            width,
            height,
            fps,
            bitrate_kbps,
            sps_pps: Vec::new(),
            encoded_any: false,
        })
    }
}

impl FrameEncoder for OpenH264Encoder {
    fn encode_frame(&mut self, frame: &Frame) -> Result<EncodedFrame> {
        let w = self.width as usize;
        let h = self.height as usize;

        // Use SIMD BGRA→YUV420 conversion (AVX2 on x86, scalar fallback).
        // ~2.8x faster than the old per-pixel f32 path (pixel_f32 callback).
        let (y, u, v) = phantom_core::color::bgra_to_yuv420(&frame.data, w, h);

        // Pack Y + U + V into a single contiguous buffer for OpenH264.
        let mut yuv_packed = Vec::with_capacity(y.len() + u.len() + v.len());
        yuv_packed.extend_from_slice(&y);
        yuv_packed.extend_from_slice(&u);
        yuv_packed.extend_from_slice(&v);

        let yuv = YUVBuffer::from_vec(yuv_packed, w, h);

        if !self.encoded_any {
            // Initialise the native encoder with this frame (output
            // discarded), tune it, then encode the frame for real as an IDR.
            self.encoder.encode(&yuv).context("H.264 encode failed")?;
            self.encoded_any = true;
            self.tune_after_init();
            self.encoder.force_intra_frame();
        }
        let bitstream = self.encoder.encode(&yuv).context("H.264 encode failed")?;
        self.encoded_any = true;
        let is_keyframe = matches!(bitstream.frame_type(), FrameType::IDR | FrameType::I);
        let mut data = bitstream.to_vec();
        if is_keyframe {
            ensure_keyframe_has_sps_pps(&mut self.sps_pps, &mut data);
        }

        Ok(EncodedFrame {
            codec: VideoCodec::H264,
            data,
            is_keyframe,
        })
    }

    fn force_keyframe(&mut self) {
        self.encoder.force_intra_frame();
    }

    fn set_bitrate_kbps(&mut self, kbps: u32) -> Result<()> {
        if kbps == self.bitrate_kbps {
            return Ok(());
        }
        // Once initialised, OpenH264 takes a new target bitrate through
        // SetOption without restarting the stream. Recreating the encoder
        // instead would force an IDR — a large frame exactly when the rate
        // controller is reacting to congestion.
        if self.encoded_any {
            let mut info = openh264_sys2::SBitrateInfo {
                iLayer: openh264_sys2::SPATIAL_LAYER_ALL,
                iBitrate: (kbps.saturating_mul(1000)).min(i32::MAX as u32) as i32,
            };
            // SAFETY: the encoder is initialised (it has encoded a frame), and
            // ENCODER_OPTION_BITRATE takes a pointer to an SBitrateInfo that
            // lives for the duration of the call.
            let rc = unsafe {
                self.encoder.raw_api().set_option(
                    openh264_sys2::ENCODER_OPTION_BITRATE,
                    std::ptr::addr_of_mut!(info).cast(),
                )
            };
            if rc == 0 {
                tracing::debug!(
                    old_kbps = self.bitrate_kbps,
                    new_kbps = kbps,
                    "OpenH264 bitrate updated in place"
                );
                self.bitrate_kbps = kbps;
                return Ok(());
            }
            tracing::warn!(
                rc,
                kbps,
                "OpenH264 SetOption(BITRATE) failed; recreating encoder"
            );
        }
        // Not initialised yet (or SetOption refused): recreate. Before the
        // first frame this costs nothing — that frame is an IDR anyway.
        let config = encoder_config(self.fps, kbps);
        let api = openh264::OpenH264API::from_source();
        self.encoder =
            Encoder::with_api_config(api, config).context("failed to recreate OpenH264 encoder")?;
        self.encoded_any = false;
        self.sps_pps.clear();
        tracing::info!(
            old_kbps = self.bitrate_kbps,
            new_kbps = kbps,
            "OpenH264 bitrate reconfigured"
        );
        self.bitrate_kbps = kbps;
        Ok(())
    }

    fn bitrate_kbps(&self) -> u32 {
        self.bitrate_kbps
    }
}

fn ensure_keyframe_has_sps_pps(cached_sps_pps: &mut Vec<u8>, data: &mut Vec<u8>) {
    let mut has_sps = false;
    let mut has_pps = false;
    let mut first_idr_start = None;

    for nal in annexb_nal_ranges(data) {
        let nal_type = data[nal.nal_start] & 0x1f;
        match nal_type {
            5 if first_idr_start.is_none() => first_idr_start = Some(nal.start_code_start),
            7 => has_sps = true,
            8 => has_pps = true,
            _ => {}
        }
    }

    if has_sps && has_pps {
        if cached_sps_pps.is_empty() {
            if let Some(idr_start) = first_idr_start {
                cached_sps_pps.extend_from_slice(&data[..idr_start]);
                tracing::debug!(
                    sps_pps_len = cached_sps_pps.len(),
                    "OpenH264: saved SPS/PPS from keyframe"
                );
            }
        }
    } else if !cached_sps_pps.is_empty() {
        let mut with_sps = Vec::with_capacity(cached_sps_pps.len() + data.len());
        with_sps.extend_from_slice(cached_sps_pps);
        with_sps.extend_from_slice(data);
        *data = with_sps;
    }
}

#[derive(Clone, Copy)]
struct AnnexBNalRange {
    start_code_start: usize,
    nal_start: usize,
}

fn annexb_nal_ranges(data: &[u8]) -> Vec<AnnexBNalRange> {
    let mut out = Vec::new();
    let mut search_from = 0usize;

    while let Some((start_code_start, start_code_len)) = find_annexb_start(data, search_from) {
        let nal_start = start_code_start + start_code_len;
        if nal_start < data.len() {
            out.push(AnnexBNalRange {
                start_code_start,
                nal_start,
            });
        }
        search_from = nal_start;
    }

    out
}

fn find_annexb_start(data: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut i = from;
    while i + 3 < data.len() {
        if data[i..].starts_with(&[0, 0, 0, 1]) {
            return Some((i, 4));
        }
        if data[i..].starts_with(&[0, 0, 1]) {
            return Some((i, 3));
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_nal(data: &[u8], nal_type: u8) -> bool {
        annexb_nal_ranges(data)
            .into_iter()
            .any(|nal| (data[nal.nal_start] & 0x1f) == nal_type)
    }

    #[test]
    fn openh264_prepends_cached_sps_pps_to_forced_keyframe() {
        let mut cache = Vec::new();
        let mut first = vec![
            0, 0, 0, 1, 0x67, 1, 2, // SPS
            0, 0, 0, 1, 0x68, 3, 4, // PPS
            0, 0, 0, 1, 0x65, 5, 6, // IDR
        ];
        ensure_keyframe_has_sps_pps(&mut cache, &mut first);
        assert_eq!(cache, vec![0, 0, 0, 1, 0x67, 1, 2, 0, 0, 0, 1, 0x68, 3, 4]);

        let mut forced = vec![0, 0, 0, 1, 0x65, 9, 10];
        ensure_keyframe_has_sps_pps(&mut cache, &mut forced);
        assert!(has_nal(&forced, 7));
        assert!(has_nal(&forced, 8));
        assert!(has_nal(&forced, 5));
    }

    fn noisy_frame(w: u32, h: u32, seed: u32) -> Frame {
        let mut x = seed.wrapping_mul(2654435761).wrapping_add(1);
        let data = (0..w * h * 4)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x as u8
            })
            .collect();
        Frame {
            width: w,
            height: h,
            format: phantom_core::frame::PixelFormat::Bgra8,
            data,
            timestamp: std::time::Instant::now(),
        }
    }

    /// Moving, compressible content (gradients + texture), unlike white noise
    /// which no QP can shrink much.
    fn moving_frame(w: u32, h: u32, i: u32) -> Frame {
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let a = (x + i * 5) ^ (y + i * 3);
                let b = ((x * 3 + y * 2 + i * 7) % 256) as u8;
                data.extend_from_slice(&[a as u8, b, (a as u8).wrapping_add(b), 255]);
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
    fn openh264_bitrate_change_is_in_place_without_idr() {
        let (w, h) = (320, 240);
        let mut enc = OpenH264Encoder::new(w, h, 30.0, 8000).unwrap();
        let first = enc.encode_frame(&moving_frame(w, h, 1)).unwrap();
        assert!(first.is_keyframe);
        let high: usize = (2..32)
            .map(|i| enc.encode_frame(&moving_frame(w, h, i)).unwrap().data.len())
            .sum();

        enc.set_bitrate_kbps(300).unwrap();
        assert!(
            enc.encoded_any,
            "bitrate change must not recreate the encoder"
        );
        let mut low = 0usize;
        for i in 32..62 {
            let f = enc.encode_frame(&moving_frame(w, h, i)).unwrap();
            assert!(!f.is_keyframe, "bitrate change produced a keyframe at {i}");
            low += f.data.len();
        }
        assert!(
            low * 3 < high,
            "lower target must shrink output: {high} bytes before, {low} after"
        );
    }

    #[test]
    fn openh264_scrolling_content_gets_no_unrequested_keyframes() {
        let (w, h) = (640u32, 360u32);
        // A tall page of text-like noise, scrolled 12 rows per frame.
        let page = noisy_frame(w, h * 8, 7).data;
        let mut enc = OpenH264Encoder::new(w, h, 30.0, 3000).unwrap();
        let mut keyframes = 0;
        for i in 0..90usize {
            let off = (i * 12 * w as usize) * 4;
            let frame = Frame {
                width: w,
                height: h,
                format: phantom_core::frame::PixelFormat::Bgra8,
                data: page[off..off + (w * h * 4) as usize].to_vec(),
                timestamp: std::time::Instant::now(),
            };
            let f = enc.encode_frame(&frame).unwrap();
            assert!(!f.data.is_empty(), "frame {i} was skipped by the encoder");
            if f.is_keyframe {
                keyframes += 1;
            }
        }
        assert_eq!(keyframes, 1, "only the first frame should be a keyframe");
    }

    #[test]
    fn openh264_sps_pps_cache_supports_three_byte_start_codes() {
        let mut cache = Vec::new();
        let mut first = vec![
            0, 0, 1, 0x67, 1, //
            0, 0, 1, 0x68, 2, //
            0, 0, 1, 0x65, 3,
        ];
        ensure_keyframe_has_sps_pps(&mut cache, &mut first);
        assert_eq!(cache, vec![0, 0, 1, 0x67, 1, 0, 0, 1, 0x68, 2]);
    }
}
