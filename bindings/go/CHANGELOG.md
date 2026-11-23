# Changelog

## 0.3.6

- Runs engine 0.3.6.

## 0.3.5

- First release. Runs Morflow pipelines from Go on Linux (x86-64 and ARM64), macOS (Apple silicon and Intel) and Windows (x86-64), with engine 0.3.5.
- No cgo: the engine library is loaded with purego, so `CGO_ENABLED=0` builds and cross-compiling work.
- The engine library is downloaded on first use, checked against the release's checksums and cached in `~/.morflow/capi`. `MORFLOW_CAPI_PATH` points at a library you provide, and `morflow-fetch` downloads it ahead of time.
