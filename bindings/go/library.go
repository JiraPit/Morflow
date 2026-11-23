package morflow

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"time"
)

// DefaultReleaseURL is where Morflow's GitHub releases are downloaded from.
// MORFLOW_RELEASE_URL replaces it with a mirror, or a folder with the same
// <tag>/<file> layout.
const DefaultReleaseURL = "https://github.com/JiraPit/Morflow/releases/download"

var checksumLine = regexp.MustCompile(`^([0-9a-f]{64}) [ *]?(.+)$`)

// parseChecksums reads a checksums.txt written by sha256sum: file -> hash.
func parseChecksums(text string) map[string]string {
	sums := map[string]string{}
	for _, line := range strings.Split(text, "\n") {
		if m := checksumLine.FindStringSubmatch(strings.TrimSpace(line)); m != nil {
			sums[m[2]] = m[1]
		}
	}
	return sums
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// releases downloads files from Morflow's releases into a cache folder.
type releases struct {
	baseURL string
	cache   string
	client  *http.Client
}

func newReleases(baseURL, cache string) *releases {
	if baseURL == "" {
		baseURL = DefaultReleaseURL
	}
	return &releases{
		baseURL: strings.TrimRight(baseURL, "/"),
		cache:   cache,
		client:  &http.Client{Timeout: 5 * time.Minute},
	}
}

// fetch returns the path of asset from release tag, saved as saveAs in the
// cache, downloading it when it is not cached yet. A cached copy is reused
// while it matches the release's checksum.
func (r *releases) fetch(tag, asset, saveAs string) (string, error) {
	dir := filepath.Join(r.cache, safeName(tag))
	file := filepath.Join(dir, saveAs)
	checksums := filepath.Join(dir, "checksums.txt")

	// Published files never change, so a cached checksum stays valid.
	if text, err := os.ReadFile(checksums); err == nil {
		if want, ok := parseChecksums(string(text))[asset]; ok {
			if data, err := os.ReadFile(file); err == nil && sha256Hex(data) == want {
				return file, nil
			}
		}
	}

	text, err := r.download(tag, "checksums.txt", "release "+tag)
	if err != nil {
		return "", err
	}
	want, ok := parseChecksums(string(text))[asset]
	if !ok {
		return "", fmt.Errorf("morflow: release %s has no %s (%s)", tag, asset, r.url(tag, ""))
	}
	data, err := r.download(tag, asset, asset+" from release "+tag)
	if err != nil {
		return "", err
	}
	if got := sha256Hex(data); got != want {
		return "", fmt.Errorf("morflow: %s from release %s does not match its checksum (expected %s, got %s); the download was corrupted or changed, so it was not used", asset, tag, want, got)
	}
	if err := os.MkdirAll(filepath.Dir(file), 0o755); err != nil {
		return "", fmt.Errorf("morflow: cannot create %s: %w", filepath.Dir(file), err)
	}
	if err := os.WriteFile(checksums, text, 0o644); err != nil {
		return "", fmt.Errorf("morflow: cannot write %s: %w", checksums, err)
	}
	// Written beside the final name and renamed, so a process that stops
	// halfway never leaves a broken library behind.
	partial, err := os.CreateTemp(filepath.Dir(file), filepath.Base(file)+".*.partial")
	if err != nil {
		return "", fmt.Errorf("morflow: cannot write in %s: %w", filepath.Dir(file), err)
	}
	_, werr := partial.Write(data)
	cerr := partial.Close()
	if werr != nil || cerr != nil {
		os.Remove(partial.Name())
		return "", fmt.Errorf("morflow: cannot write %s: %w", file, errors.Join(werr, cerr))
	}
	if err := os.Rename(partial.Name(), file); err != nil {
		os.Remove(partial.Name())
		// Another process may have put the same file there first.
		if data, rerr := os.ReadFile(file); rerr == nil && sha256Hex(data) == want {
			return file, nil
		}
		return "", fmt.Errorf("morflow: cannot write %s: %w", file, err)
	}
	return file, nil
}

func (r *releases) url(tag, asset string) string {
	return r.baseURL + "/" + tag + "/" + asset
}

func (r *releases) download(tag, asset, what string) ([]byte, error) {
	location := r.url(tag, asset)
	if !strings.HasPrefix(location, "http://") && !strings.HasPrefix(location, "https://") {
		// A file: URL or a plain folder.
		path := location
		if strings.HasPrefix(location, "file:") {
			u, err := url.Parse(location)
			if err != nil {
				return nil, fmt.Errorf("morflow: bad release URL %s: %w", location, err)
			}
			path = fileURLPath(u)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("morflow: could not find %s: %w", what, err)
		}
		return data, nil
	}
	response, err := r.client.Get(location)
	if err != nil {
		return nil, fmt.Errorf("morflow: could not download %s from %s: %w", what, location, err)
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusNotFound {
		return nil, fmt.Errorf("morflow: could not find %s: %s returned 404; is the version published?", what, location)
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("morflow: could not download %s: %s returned HTTP %d", what, location, response.StatusCode)
	}
	data, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("morflow: could not download %s from %s: %w", what, location, err)
	}
	return data, nil
}

var windowsDrive = regexp.MustCompile(`^/?[A-Za-z]:`)

// fileURLPath is the local path a file: URL names. On Windows that is
// file:///C:/releases, whose URL path is /C:/releases; file://C:/releases,
// which puts the drive where the host goes, is accepted too.
func fileURLPath(u *url.URL) string {
	path := u.Path
	if windowsDrive.MatchString(u.Host) {
		path = u.Host + path
	}
	if windowsDrive.MatchString(path) && strings.HasPrefix(path, "/") {
		path = path[1:]
	}
	return filepath.FromSlash(path)
}

var unsafeChars = regexp.MustCompile(`[^A-Za-z0-9._-]`)

// Tags such as action_packs/base/v0.3.4 become one folder name.
func safeName(name string) string { return unsafeChars.ReplaceAllString(name, "_") }

// cacheDir is where downloaded engine libraries are kept: ~/.morflow/capi,
// beside the action cache.
func cacheDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("morflow: no home folder to cache the engine library in; set MORFLOW_CAPI_PATH: %w", err)
	}
	return filepath.Join(home, ".morflow", "capi"), nil
}

// FetchLibrary downloads the engine library for this platform into the
// cache, if it is not there already, and returns its path. Load and FromStr
// do this themselves; call it ahead of time to download while building a
// container image, or to fail early without a network.
func FetchLibrary() (string, error) {
	return fetchLibrary(runtime.GOOS, runtime.GOARCH)
}

func fetchLibrary(goos, goarch string) (string, error) {
	platform, err := platformName(goos, goarch)
	if err != nil {
		return "", err
	}
	cache, err := cacheDir()
	if err != nil {
		return "", err
	}
	r := newReleases(os.Getenv("MORFLOW_RELEASE_URL"), cache)
	return r.fetch("v"+EngineVersion, capiAsset(platform), filepath.Join(platform, libraryFile(platform)))
}

// libraryPath is the engine library to load: MORFLOW_CAPI_PATH, or the
// downloaded and cached one.
func libraryPath() (string, error) {
	if path := os.Getenv("MORFLOW_CAPI_PATH"); path != "" {
		if _, err := os.Stat(path); err != nil {
			return "", fmt.Errorf("morflow: MORFLOW_CAPI_PATH is %s, which cannot be used: %w", path, err)
		}
		return path, nil
	}
	return FetchLibrary()
}
