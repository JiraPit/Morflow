// Bind to the API exposed by the selected shared system OpenCV installation.
fn main() {
    println!("cargo:rustc-check-cfg=cfg(opencv_gaussian_hint)");
    for variable in [
        "OPENCV_INCLUDE_PATHS",
        "OPENCV_LINK_PATHS",
        "OPENCV_LINK_LIBS",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    let probe = if std::env::var("OPENCV_INCLUDE_PATHS").is_err()
        || std::env::var("OPENCV_LINK_PATHS").is_err()
    {
        Some(pkg_config::Config::new().cargo_metadata(false).probe("opencv4").expect("Set OpenCV include/link paths or install OpenCV development pkg-config metadata"))
    } else {
        None
    };
    let paths: Vec<std::path::PathBuf> = std::env::var("OPENCV_INCLUDE_PATHS")
        .map(|paths| paths.split(',').map(Into::into).collect())
        .unwrap_or_else(|_| probe.as_ref().unwrap().include_paths.clone());
    let link_paths: Vec<std::path::PathBuf> = std::env::var("OPENCV_LINK_PATHS")
        .map(|paths| paths.split(',').map(Into::into).collect())
        .unwrap_or_else(|_| probe.as_ref().unwrap().link_paths.clone());
    let libraries = std::env::var("OPENCV_LINK_LIBS")
        .unwrap_or_else(|_| "dylib=opencv_core,dylib=opencv_imgproc".into());
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    for library in libraries.split(',') {
        let name = library
            .strip_prefix("dylib=")
            .expect("OpenCV must use explicitly shared dylib=<name> libraries");
        let exists = link_paths.iter().any(|path| match target_os.as_str() {
            "windows" => {
                path.join(format!("{name}.dll")).exists()
                    || path.join("../bin").join(format!("{name}.dll")).exists()
            }
            "macos" => path.join(format!("lib{name}.dylib")).exists(),
            _ => path.join(format!("lib{name}.so")).exists(),
        });
        assert!(exists,"Shared OpenCV library {name} is missing from the selected installation; static libraries are not supported");
    }
    let header = paths
        .iter()
        .map(|p| p.join("opencv2/imgproc.hpp"))
        .find(|p| p.exists())
        .expect("OpenCV imgproc headers are required to build this plugin");
    println!("cargo:rerun-if-changed={}", header.display());
    let text = std::fs::read_to_string(header).expect("Read OpenCV imgproc header");
    let declaration = text
        .split("void GaussianBlur(")
        .nth(1)
        .expect("GaussianBlur declaration")
        .split(';')
        .next()
        .unwrap();
    if declaration.contains("AlgorithmHint") {
        println!("cargo:rustc-cfg=opencv_gaussian_hint");
    }
}
