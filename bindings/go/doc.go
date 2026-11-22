// Package morflow runs Morflow pipelines (.morf files) from Go: image, audio
// and tensor processing in native code, the same as in Python, Node.js and
// the other Morflow packages.
//
// Prepare the actions a pipeline uses once with the Morflow CLI, which fills
// the action cache in ~/.morflow/actions:
//
//	morflow prep pipelines/clean.morf
//
// Then compile and run the pipeline:
//
//	pipeline, err := morflow.Load("pipelines/clean.morf")
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer pipeline.Close()
//
//	out, err := pipeline.Run(wav) // wav is a []byte
//	if err != nil {
//		log.Fatal(err)
//	}
//	os.WriteFile("clean.wav", out.Bytes(), 0o644)
//
// The package calls Morflow's engine library, libmorflow_capi, without cgo.
// The first Load or FromStr downloads the library for this platform from
// Morflow's GitHub release, checks it against the release's checksums and
// caches it in ~/.morflow/capi, so later runs work offline. Set
// MORFLOW_CAPI_PATH to use a library file you provide instead, for example
// one fetched ahead of time with the morflow-fetch command.
//
// Pipelines are safe to use from many goroutines at once.
package morflow
