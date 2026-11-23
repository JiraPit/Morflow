package morflow

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
)

func TestPlatformNames(t *testing.T) {
	for _, c := range []struct{ goos, goarch, platform, file, asset string }{
		{"linux", "amd64", "linux-x86_64", "libmorflow_capi.so", "libmorflow_capi-linux-x86_64.so"},
		{"linux", "arm64", "linux-aarch64", "libmorflow_capi.so", "libmorflow_capi-linux-aarch64.so"},
		{"darwin", "arm64", "darwin-arm64", "libmorflow_capi.dylib", "libmorflow_capi-darwin-arm64.dylib"},
		{"darwin", "amd64", "darwin-x86_64", "libmorflow_capi.dylib", "libmorflow_capi-darwin-x86_64.dylib"},
		{"windows", "amd64", "windows-x86_64", "morflow_capi.dll", "libmorflow_capi-windows-x86_64.dll"},
	} {
		platform, err := platformName(c.goos, c.goarch)
		if err != nil || platform != c.platform {
			t.Fatalf("%s/%s: got %q, %v", c.goos, c.goarch, platform, err)
		}
		if libraryFile(platform) != c.file || capiAsset(platform) != c.asset {
			t.Fatalf("%s: got %s, %s", platform, libraryFile(platform), capiAsset(platform))
		}
	}
	for _, c := range [][2]string{{"linux", "386"}, {"windows", "arm64"}, {"freebsd", "amd64"}, {"js", "wasm"}} {
		if _, err := platformName(c[0], c[1]); err == nil || !strings.Contains(err.Error(), "no engine library") {
			t.Fatalf("%s/%s: %v", c[0], c[1], err)
		}
	}
}

func TestParseChecksums(t *testing.T) {
	a, b := strings.Repeat("a", 64), strings.Repeat("b", 64)
	sums := parseChecksums(a + "  one.so\n" + b + " *two.dll\r\nnot a line\n\n" + strings.Repeat("c", 63) + "  short\n")
	if len(sums) != 2 || sums["one.so"] != a || sums["two.dll"] != b {
		t.Fatalf("got %v", sums)
	}
	if safeName("action_packs/base/v0.3.4") != "action_packs_base_v0.3.4" {
		t.Fatal(safeName("action_packs/base/v0.3.4"))
	}
}

// publish lays out a release like GitHub's: <root>/<tag>/<asset> and a
// checksums.txt. checksum overrides the asset's real hash.
func publish(t *testing.T, root, tag, asset string, data []byte, checksum string) {
	t.Helper()
	dir := filepath.Join(root, filepath.FromSlash(tag))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if checksum == "" {
		checksum = sha256Hex(data)
	}
	os.WriteFile(filepath.Join(dir, asset), data, 0o644)
	os.WriteFile(filepath.Join(dir, "checksums.txt"), []byte(checksum+"  "+asset+"\n"), 0o644)
}

// fileURL is the file: URL of a local folder, file:///C:/… on Windows.
func fileURL(path string) string {
	path = filepath.ToSlash(path)
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	return (&url.URL{Scheme: "file", Path: path}).String()
}

func TestFileURLPaths(t *testing.T) {
	for raw, want := range map[string]string{
		"file:///srv/releases":     "/srv/releases",
		"file:///C:/releases":      "C:/releases",
		"file://C:/releases":       "C:/releases",
		"file:///c:/Users/a%20b/x": "c:/Users/a b/x",
	} {
		u, err := url.Parse(raw)
		if err != nil {
			t.Fatal(err)
		}
		if got := fileURLPath(u); got != filepath.FromSlash(want) {
			t.Fatalf("%s: got %s, want %s", raw, got, filepath.FromSlash(want))
		}
	}
}

func TestFetchFromAFolderAndCache(t *testing.T) {
	root, cache := t.TempDir(), t.TempDir()
	publish(t, root, "v1.0.0", "lib.so", []byte("library"), "")

	for _, base := range []string{root, fileURL(root)} {
		cache := filepath.Join(cache, safeName(base))
		r := newReleases(base, cache)
		file, err := r.fetch("v1.0.0", "lib.so", filepath.Join("linux-x86_64", "lib.so"))
		if err != nil {
			t.Fatalf("%s: %v", base, err)
		}
		if data, _ := os.ReadFile(file); string(data) != "library" {
			t.Fatalf("%s: got %q", base, data)
		}
		if filepath.Dir(file) != filepath.Join(cache, "v1.0.0", "linux-x86_64") {
			t.Fatalf("saved at %s", file)
		}
	}

	// Cached: works with the release gone.
	r := newReleases(root, filepath.Join(cache, safeName(root)))
	os.RemoveAll(filepath.Join(root, "v1.0.0"))
	if _, err := r.fetch("v1.0.0", "lib.so", filepath.Join("linux-x86_64", "lib.so")); err != nil {
		t.Fatalf("cached copy: %v", err)
	}
	// A damaged cached copy is not used.
	cached := filepath.Join(cache, safeName(root), "v1.0.0", "linux-x86_64", "lib.so")
	os.WriteFile(cached, []byte("damaged"), 0o644)
	if _, err := r.fetch("v1.0.0", "lib.so", filepath.Join("linux-x86_64", "lib.so")); err == nil || !strings.Contains(err.Error(), "could not find release v1.0.0") {
		t.Fatalf("damaged cache with no release: %v", err)
	}
	// It is replaced once the release is reachable again.
	publish(t, root, "v1.0.0", "lib.so", []byte("library"), "")
	file, err := r.fetch("v1.0.0", "lib.so", filepath.Join("linux-x86_64", "lib.so"))
	if data, _ := os.ReadFile(file); err != nil || string(data) != "library" {
		t.Fatalf("repaired: %q, %v", data, err)
	}
}

func TestFetchRejectsBadReleases(t *testing.T) {
	root := t.TempDir()
	publish(t, root, "v1.0.0", "lib.so", []byte("changed"), strings.Repeat("0", 64))
	r := newReleases(root, t.TempDir())
	for _, c := range []struct{ tag, asset, message string }{
		{"v1.0.0", "lib.so", "does not match its checksum"},
		{"v1.0.0", "other.so", "release v1.0.0 has no other.so"},
		{"v9.9.9", "lib.so", "could not find release v9.9.9"},
	} {
		if _, err := r.fetch(c.tag, c.asset, c.asset); err == nil || !strings.Contains(err.Error(), c.message) {
			t.Fatalf("%s/%s: %v", c.tag, c.asset, err)
		}
	}
	// A checksums.txt that lists a file that is not there.
	publish(t, root, "v2.0.0", "lib.so", []byte("x"), "")
	os.Remove(filepath.Join(root, "v2.0.0", "lib.so"))
	if _, err := r.fetch("v2.0.0", "lib.so", "lib.so"); err == nil || !strings.Contains(err.Error(), "could not find lib.so") {
		t.Fatalf("missing asset: %v", err)
	}
	if _, err := newReleases("file://%zz", t.TempDir()).fetch("v1", "a", "a"); err == nil || !strings.Contains(err.Error(), "bad release URL") {
		t.Fatalf("bad URL: %v", err)
	}
}

func TestFetchOverHTTP(t *testing.T) {
	library := []byte("engine library bytes")
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		switch r.URL.Path {
		case "/v1.0.0/checksums.txt":
			w.Write([]byte(sha256Hex(library) + "  lib.so\n"))
		case "/v1.0.0/lib.so":
			w.Write(library)
		case "/v5.0.0/checksums.txt":
			http.Error(w, "broken", http.StatusInternalServerError)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	r := newReleases(server.URL+"/", t.TempDir())
	file, err := r.fetch("v1.0.0", "lib.so", "lib.so")
	if err != nil {
		t.Fatal(err)
	}
	if data, _ := os.ReadFile(file); string(data) != string(library) {
		t.Fatalf("got %q", data)
	}
	before := requests.Load()
	if _, err := r.fetch("v1.0.0", "lib.so", "lib.so"); err != nil || requests.Load() != before {
		t.Fatalf("the cached copy should be used without requests: %v, %d requests", err, requests.Load()-before)
	}
	if _, err := r.fetch("v2.0.0", "lib.so", "lib.so"); err == nil || !strings.Contains(err.Error(), "returned 404") {
		t.Fatalf("404: %v", err)
	}
	if _, err := r.fetch("v5.0.0", "lib.so", "lib.so"); err == nil || !strings.Contains(err.Error(), "HTTP 500") {
		t.Fatalf("500: %v", err)
	}
	server.Close()
	if _, err := newReleases(server.URL, t.TempDir()).fetch("v1.0.0", "lib.so", "lib.so"); err == nil || !strings.Contains(err.Error(), "could not download") {
		t.Fatalf("unreachable: %v", err)
	}
}

func TestFetchCannotWrite(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("file permissions work differently on Windows")
	}
	root := t.TempDir()
	publish(t, root, "v1.0.0", "lib.so", []byte("library"), "")
	// The cache is a file, so nothing can be created inside it.
	cache := filepath.Join(t.TempDir(), "cache")
	os.WriteFile(cache, nil, 0o644)
	if _, err := newReleases(root, cache).fetch("v1.0.0", "lib.so", "lib.so"); err == nil || !strings.Contains(err.Error(), "cannot") {
		t.Fatalf("unwritable cache: %v", err)
	}
}

func TestLibraryPath(t *testing.T) {
	file := filepath.Join(t.TempDir(), "libmorflow_capi.so")
	os.WriteFile(file, []byte("x"), 0o644)
	t.Setenv("MORFLOW_CAPI_PATH", file)
	if path, err := libraryPath(); err != nil || path != file {
		t.Fatalf("got %s, %v", path, err)
	}
	t.Setenv("MORFLOW_CAPI_PATH", file+".missing")
	if _, err := libraryPath(); err == nil || !strings.Contains(err.Error(), "MORFLOW_CAPI_PATH") {
		t.Fatalf("missing file: %v", err)
	}

	// Without MORFLOW_CAPI_PATH the library comes from MORFLOW_RELEASE_URL
	// into ~/.morflow/capi.
	t.Setenv("MORFLOW_CAPI_PATH", "")
	home, releases := t.TempDir(), t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("MORFLOW_RELEASE_URL", releases)
	platform, err := platformName(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		t.Skip(err)
	}
	publish(t, releases, "v"+EngineVersion, capiAsset(platform), []byte("engine"), "")
	path, err := libraryPath()
	want := filepath.Join(home, ".morflow", "capi", "v"+EngineVersion, platform, libraryFile(platform))
	if err != nil || path != want {
		t.Fatalf("got %s, %v; want %s", path, err, want)
	}
	if _, err := fetchLibrary("plan9", "amd64"); err == nil {
		t.Fatal("an unsupported platform should fail")
	}
}

func TestNoHomeFolder(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Windows finds the home folder without HOME")
	}
	t.Setenv("HOME", "")
	t.Setenv("MORFLOW_CAPI_PATH", "")
	if _, err := FetchLibrary(); err == nil || !strings.Contains(err.Error(), "no home folder") {
		t.Fatalf("got %v", err)
	}
}

func TestFetchWriteFailures(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("file permissions work differently on Windows")
	}
	root := t.TempDir()
	publish(t, root, "v1.0.0", "lib.so", []byte("library"), "")

	// checksums.txt cannot be written: a folder has its name.
	cache := t.TempDir()
	os.MkdirAll(filepath.Join(cache, "v1.0.0", "checksums.txt"), 0o755)
	if _, err := newReleases(root, cache).fetch("v1.0.0", "lib.so", "lib.so"); err == nil || !strings.Contains(err.Error(), "cannot write") {
		t.Fatalf("checksums: %v", err)
	}

	// The library's folder is read-only.
	cache = t.TempDir()
	locked := filepath.Join(cache, "v1.0.0", "linux-x86_64")
	os.MkdirAll(locked, 0o755)
	os.Chmod(locked, 0o555)
	defer os.Chmod(locked, 0o755)
	if _, err := newReleases(root, cache).fetch("v1.0.0", "lib.so", filepath.Join("linux-x86_64", "lib.so")); err == nil || !strings.Contains(err.Error(), "cannot write in") {
		t.Fatalf("read-only folder: %v", err)
	}

	// The library's name is taken by a folder, so it cannot be renamed into
	// place, and what is there is not the library.
	cache = t.TempDir()
	os.MkdirAll(filepath.Join(cache, "v1.0.0", "lib.so", "inside"), 0o755)
	if _, err := newReleases(root, cache).fetch("v1.0.0", "lib.so", "lib.so"); err == nil || !strings.Contains(err.Error(), "cannot write") {
		t.Fatalf("name taken: %v", err)
	}
	leftovers, _ := filepath.Glob(filepath.Join(cache, "v1.0.0", "*.partial"))
	if len(leftovers) != 0 {
		t.Fatalf("partial files left behind: %v", leftovers)
	}
}

func TestFetchDroppedConnection(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Promise more than is sent, then hang up.
		w.Header().Set("Content-Length", "1000")
		w.Write([]byte("short"))
	}))
	defer server.Close()
	if _, err := newReleases(server.URL, t.TempDir()).fetch("v1.0.0", "lib.so", "lib.so"); err == nil || !strings.Contains(err.Error(), "could not download") {
		t.Fatalf("got %v", err)
	}
}
