//! Build script: only does work for the optional `x264` feature, which
//! compiles `csrc/x264_shim.c` against a static libx264 found in `$X264_DIR`
//! (default `<workspace>/target/x264`, as installed by scripts/build-x264.sh).

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "x264")]
    build_x264();
}

#[cfg(feature = "x264")]
fn build_x264() {
    use std::path::PathBuf;
    println!("cargo:rerun-if-env-changed=X264_DIR");
    println!("cargo:rerun-if-changed=csrc/x264_shim.c");
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let dir = std::env::var_os("X264_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("../../target/x264"));
    let include = dir.join("include");
    let lib = dir.join("lib");
    if !include.join("x264.h").exists() {
        panic!(
            "feature `x264`: no x264.h under {}. Run scripts/build-x264.sh or set X264_DIR.",
            include.display()
        );
    }
    cc::Build::new()
        .file("csrc/x264_shim.c")
        .include(&include)
        .warnings(true)
        .compile("phantom_x264_shim");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=static=x264");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=dl");
    }
}
