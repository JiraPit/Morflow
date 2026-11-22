# Morflow for Go

Run [Morflow](https://morflow.org) pipelines from Go: clean up audio, prepare images and turn data into tensors in native code, in servers, command-line tools and desktop programs. The same `.morf` file gives the same output here as in Python, Node.js and the other Morflow packages.

It runs on Linux (x86-64 and ARM64), macOS (Apple silicon and Intel) and Windows (x86-64), and needs no cgo.

## Install

```
go get github.com/JiraPit/Morflow/bindings/go
```

You also need the [Morflow CLI](https://morflow.org/doc/#cli-installation) to download the actions your pipelines use.

## Use it

Write a pipeline, for example `pipelines/louder.morf`:

```
import audio_basics/latest

accept Bytes $wav
$wav >> to_audio >> gain(linear=2.0) >> to_wav >> emit
```

Download its actions once:

```
morflow prep pipelines/louder.morf
```

Then run it from Go:

```go
package main

import (
	"log"
	"os"

	morflow "github.com/JiraPit/Morflow/bindings/go"
)

func main() {
	pipeline, err := morflow.Load("pipelines/louder.morf")
	if err != nil {
		log.Fatal(err)
	}
	defer pipeline.Close()

	wav, err := os.ReadFile("in.wav")
	if err != nil {
		log.Fatal(err)
	}
	out, err := pipeline.Run(wav)
	if err != nil {
		log.Fatal(err)
	}
	os.WriteFile("out.wav", out.Bytes(), 0o644)
}
```

A pipeline can run from many goroutines at once.

### Inputs

- A numeric slice (`[]float32`, `[]float64`, `[]int8`, `[]int16`, `[]int32`, `[]int64`, `[]int`, `[]uint32`, `[]uint64`): a one-dimensional tensor of that type.
- A `morflow.TensorInput` with a `Shape`, for tensors of any shape. `morflow.Image` and `morflow.Audio` build image and audio inputs.
- A `[]byte`: raw bytes, such as a WAV, PNG or JPEG file.
- A `*morflow.Tensor` from an earlier run.
- A `[]any` of those: a Composite.

### Outputs

`Run` returns a `*morflow.Tensor` with its `Shape` and `DType`. Read it with `Float32s()`, `Int64s()`, `Bytes()` and so on, or `Values()` for the slice that matches its dtype. A Composite has its parts in `Items`. Pipelines with several `emit("name")` outputs use `RunToMap`, and `RunArgs` passes the parameters declared with `accept` in order: numbers, strings and bools, or any input.

Engine errors are `*morflow.Error` values with the engine's message.

## The engine library

The first `Load` or `FromStr` downloads Morflow's engine library for your platform (about 3 MB) from the engine's GitHub release, checks it against the release's checksums, and caches it in `~/.morflow/capi`. Later runs use the cached copy, offline.

For a machine that shouldn't download on first use, such as a container image or a server without internet access, download the library ahead of time and point `MORFLOW_CAPI_PATH` at it:

```
go run github.com/JiraPit/Morflow/bindings/go/cmd/morflow-fetch@latest -out /opt/morflow
export MORFLOW_CAPI_PATH=/opt/morflow/libmorflow_capi.so
```

`MORFLOW_RELEASE_URL` downloads from a mirror instead of GitHub, or from a folder with the same layout as the releases.

## More

- [Morflow in Go](https://morflow.org/doc/#exec-go)
- [Pipeline syntax](https://morflow.org/doc/#syntax)
- [Report a problem](https://github.com/JiraPit/Morflow/issues)
