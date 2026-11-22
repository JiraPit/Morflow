package morflow

import (
	"fmt"
	"strings"
)

// platformName is Morflow's name for a platform, as its releases use it
// (scripts/native_targets.py), such as "linux-x86_64".
func platformName(goos, goarch string) (string, error) {
	arch, ok := map[string]string{"amd64": "x86_64", "arm64": "aarch64"}[goarch]
	if !ok {
		return "", fmt.Errorf("morflow: no engine library for %s/%s; Morflow runs on amd64 and arm64", goos, goarch)
	}
	switch {
	case goos == "linux":
		return "linux-" + arch, nil
	case goos == "darwin" && goarch == "arm64":
		return "darwin-arm64", nil
	case goos == "darwin":
		return "darwin-x86_64", nil
	case goos == "windows" && goarch == "amd64":
		return "windows-x86_64", nil
	}
	return "", fmt.Errorf("morflow: no engine library for %s/%s; Morflow runs on Linux (amd64, arm64), macOS (arm64, amd64) and Windows (amd64)", goos, goarch)
}

// extension is the library file extension on a platform.
func extension(platform string) string {
	switch {
	case strings.HasPrefix(platform, "windows-"):
		return "dll"
	case strings.HasPrefix(platform, "darwin-"):
		return "dylib"
	}
	return "so"
}

// libraryFile is the engine library's file name on a platform, such as
// libmorflow_capi.so or morflow_capi.dll.
func libraryFile(platform string) string {
	if extension(platform) == "dll" {
		return "morflow_capi.dll"
	}
	return "libmorflow_capi." + extension(platform)
}

// capiAsset is the engine library's file name in a GitHub release.
func capiAsset(platform string) string {
	return "libmorflow_capi-" + platform + "." + extension(platform)
}
