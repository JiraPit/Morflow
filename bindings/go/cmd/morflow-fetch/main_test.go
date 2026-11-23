package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	morflow "github.com/JiraPit/Morflow/bindings/go"
)

// release lays out a fake engine release in a folder and points
// MORFLOW_RELEASE_URL and the home folder at temporary places.
func release(t *testing.T) (asset string) {
	t.Helper()
	platform := map[string]string{
		"linux/amd64": "linux-x86_64", "linux/arm64": "linux-aarch64",
		"darwin/arm64": "darwin-arm64", "darwin/amd64": "darwin-x86_64",
		"windows/amd64": "windows-x86_64",
	}[runtime.GOOS+"/"+runtime.GOARCH]
	if platform == "" {
		t.Skip("no engine library for this platform")
	}
	ext := map[string]string{"linux": "so", "darwin": "dylib", "windows": "dll"}[runtime.GOOS]
	asset = "libmorflow_capi-" + platform + "." + ext
	root := t.TempDir()
	dir := filepath.Join(root, "v"+morflow.EngineVersion)
	os.MkdirAll(dir, 0o755)
	data := []byte("engine library")
	sum := sha256.Sum256(data)
	os.WriteFile(filepath.Join(dir, asset), data, 0o644)
	os.WriteFile(filepath.Join(dir, "checksums.txt"), []byte(hex.EncodeToString(sum[:])+"  "+asset+"\n"), 0o644)
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("MORFLOW_RELEASE_URL", root)
	return asset
}

func TestFetchIntoTheCache(t *testing.T) {
	release(t)
	var out bytes.Buffer
	if err := run(nil, &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), filepath.Join(".morflow", "capi")) {
		t.Fatalf("got %q", out.String())
	}
}

func TestFetchIntoAFolder(t *testing.T) {
	release(t)
	folder := filepath.Join(t.TempDir(), "lib")
	var out bytes.Buffer
	if err := run([]string{"-out", folder}, &out); err != nil {
		t.Fatal(err)
	}
	entries, _ := os.ReadDir(folder)
	if len(entries) != 1 || !strings.Contains(out.String(), folder) {
		t.Fatalf("got %v, %q", entries, out.String())
	}
	data, _ := os.ReadFile(filepath.Join(folder, entries[0].Name()))
	if string(data) != "engine library" {
		t.Fatalf("copied %q", data)
	}
}

func TestFetchErrors(t *testing.T) {
	release(t)
	var out bytes.Buffer
	if err := run([]string{"extra"}, &out); err == nil || !strings.Contains(err.Error(), "takes no arguments") {
		t.Fatalf("extra argument: %v", err)
	}
	if err := run([]string{"-nope"}, &out); err == nil {
		t.Fatal("an unknown flag should fail")
	}
	blocker := filepath.Join(t.TempDir(), "file")
	os.WriteFile(blocker, nil, 0o644)
	if err := run([]string{"-out", filepath.Join(blocker, "lib")}, &out); err == nil {
		t.Fatal("a folder inside a file should fail")
	}
	// With nothing cached yet, a missing release fails.
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("MORFLOW_RELEASE_URL", t.TempDir())
	if err := run(nil, &out); err == nil {
		t.Fatal("a missing release should fail")
	}
}

func TestFetchIntoAReadOnlyFolder(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("file permissions work differently on Windows")
	}
	release(t)
	folder := t.TempDir()
	os.Chmod(folder, 0o555)
	defer os.Chmod(folder, 0o755)
	if err := run([]string{"-out", folder}, &bytes.Buffer{}); err == nil {
		t.Fatal("a read-only folder should fail")
	}
}
