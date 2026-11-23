// Command morflow-fetch downloads Morflow's engine library ahead of time, for
// machines that should not download it on first use, such as a container
// image or a server without internet access.
//
//	go run github.com/JiraPit/Morflow/bindings/go/cmd/morflow-fetch@latest
//
// downloads the library into the cache (~/.morflow/capi), where the morflow
// package finds it. With -out, it also copies the library into a folder and
// prints its path; point MORFLOW_CAPI_PATH at that file:
//
//	go run github.com/JiraPit/Morflow/bindings/go/cmd/morflow-fetch@latest -out /opt/morflow
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	morflow "github.com/JiraPit/Morflow/bindings/go"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string, stdout io.Writer) error {
	flags := flag.NewFlagSet("morflow-fetch", flag.ContinueOnError)
	out := flags.String("out", "", "also copy the library into this folder")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() > 0 {
		return fmt.Errorf("morflow-fetch takes no arguments, only -out <folder>")
	}
	path, err := morflow.FetchLibrary()
	if err != nil {
		return err
	}
	if *out != "" {
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(*out, 0o755); err != nil {
			return err
		}
		path = filepath.Join(*out, filepath.Base(path))
		if err := os.WriteFile(path, data, 0o755); err != nil {
			return err
		}
	}
	fmt.Fprintf(stdout, "Morflow engine %s: %s\n", morflow.EngineVersion, path)
	return nil
}
