// Command example doubles the volume of a WAV file with a Morflow pipeline.
//
//	morflow prep example/louder.morf
//	go run ./example in.wav out.wav
package main

import (
	"fmt"
	"os"

	morflow "github.com/JiraPit/Morflow/bindings/go"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: example <in.wav> <out.wav>")
		os.Exit(2)
	}
	if err := louder(os.Args[1], os.Args[2]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func louder(in, out string) error {
	pipeline, err := morflow.Load("example/louder.morf")
	if err != nil {
		return err
	}
	defer pipeline.Close()

	wav, err := os.ReadFile(in)
	if err != nil {
		return err
	}
	result, err := pipeline.Run(wav)
	if err != nil {
		return err
	}
	return os.WriteFile(out, result.Bytes(), 0o644)
}
