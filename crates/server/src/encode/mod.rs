//! Encoders that run on the CPU. GPU-backed encoders (NVENC, NVDEC) live
//! in the phantom-gpu crate so they can be loaded via runtime dlopen
//! without forcing CUDA on every build.

pub mod h264;

/// x264 (optional `x264` feature, GPL).
#[cfg(feature = "x264")]
pub mod x264;
