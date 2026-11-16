//! The platform this engine was compiled for, as used in artifact names.

/// The host platform name and its native library extension.
pub(crate) fn host_platform() -> (&'static str, &'static str) {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        ("linux-x86_64", "so")
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        ("linux-aarch64", "so")
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        ("darwin-arm64", "dylib")
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        ("darwin-x86_64", "dylib")
    }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        ("windows-x86_64", "dll")
    }
    #[cfg(all(target_os = "android", target_arch = "aarch64"))]
    {
        ("android-aarch64", "so")
    }
    #[cfg(all(target_os = "android", target_arch = "x86_64"))]
    {
        ("android-x86_64", "so")
    }
    #[cfg(all(target_os = "ios", target_arch = "aarch64", not(target_abi = "sim")))]
    {
        ("ios-arm64", "dylib")
    }
    #[cfg(all(target_os = "ios", target_arch = "aarch64", target_abi = "sim"))]
    {
        ("ios-arm64-sim", "dylib")
    }
    #[cfg(not(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "android", target_arch = "aarch64"),
        all(target_os = "android", target_arch = "x86_64"),
        all(target_os = "ios", target_arch = "aarch64")
    )))]
    {
        ("unsupported", "bin")
    }
}

/// The native library extension used by artifacts for `platform`.
pub(crate) fn library_extension(platform: &str) -> &'static str {
    if platform.starts_with("windows-") {
        "dll"
    } else if platform.starts_with("darwin-") || platform.starts_with("ios-") {
        "dylib"
    } else {
        "so"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_platform_is_supported_and_consistent() {
        let (platform, extension) = host_platform();
        assert_ne!(platform, "unsupported");
        assert_eq!(library_extension(platform), extension);
    }

    #[test]
    fn extensions_follow_the_platform_family() {
        for (platform, extension) in [
            ("linux-x86_64", "so"),
            ("android-aarch64", "so"),
            ("android-x86_64", "so"),
            ("darwin-arm64", "dylib"),
            ("ios-arm64", "dylib"),
            ("ios-arm64-sim", "dylib"),
            ("windows-x86_64", "dll"),
        ] {
            assert_eq!(library_extension(platform), extension, "{platform}");
        }
    }
}
