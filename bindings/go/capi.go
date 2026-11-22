package morflow

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Value kinds of MorflowValueIn (bindings/capi/src/lib.rs).
const (
	kindTensor uint32 = 0
	kindBytes  uint32 = 1
	kindNumber uint32 = 2
	kindText   uint32 = 3
	kindBool   uint32 = 4
)

// valueIn mirrors MorflowValueIn: a value passed into a pipeline. Unused
// pointers are nil, and sampleRate and channels are -1 when absent.
type valueIn struct {
	kind        uint32
	data        unsafe.Pointer
	dataLen     uintptr
	shape       unsafe.Pointer
	rank        uintptr
	dtype       unsafe.Pointer
	payloadType unsafe.Pointer
	colorSpace  unsafe.Pointer
	layout      unsafe.Pointer
	sampleRate  int64
	channels    int64
	items       unsafe.Pointer
	itemCount   uintptr
	number      float64
	flag        uint8
}

// valueOut mirrors MorflowValueOut: an emitted value, in engine memory.
type valueOut struct {
	dtype     *byte
	shape     unsafe.Pointer
	rank      uintptr
	data      unsafe.Pointer
	dataLen   uintptr
	items     unsafe.Pointer
	itemCount uintptr
}

// capi holds the engine library's functions.
type capi struct {
	version      func() *byte
	load         func(path *byte, errOut *unsafe.Pointer) unsafe.Pointer
	fromStr      func(source *byte, errOut *unsafe.Pointer) unsafe.Pointer
	pipelineFree func(pipeline unsafe.Pointer)
	paramCount   func(pipeline unsafe.Pointer) uintptr
	paramName    func(pipeline unsafe.Pointer, index uintptr) *byte
	stringFree   func(text *byte)
	run          func(pipeline unsafe.Pointer, input *valueIn, all uint8) unsafe.Pointer
	runArgs      func(pipeline unsafe.Pointer, args *valueIn, count uintptr, all uint8) unsafe.Pointer
	resultError  func(result unsafe.Pointer) *byte
	resultCount  func(result unsafe.Pointer) uintptr
	resultName   func(result unsafe.Pointer, index uintptr) *byte
	resultValue  func(result unsafe.Pointer, index uintptr) *valueOut
	resultFree   func(result unsafe.Pointer)
}

// expectedVersion is the engine version the library must report; tests
// change it to check the mismatch error.
var expectedVersion = EngineVersion

var engine struct {
	sync.Mutex
	api *capi
}

// loadEngine loads the engine library once per process. A failure is not
// remembered, so a later call can succeed, for example once the network is
// back.
func loadEngine() (*capi, error) {
	engine.Lock()
	defer engine.Unlock()
	if engine.api != nil {
		return engine.api, nil
	}
	path, err := libraryPath()
	if err != nil {
		return nil, err
	}
	api, err := bindLibrary(path)
	if err != nil {
		return nil, err
	}
	if native := goString(api.version()); native != expectedVersion {
		return nil, fmt.Errorf("morflow: the engine library %s is version %s, but this package runs engine %s; use the library from release v%s", path, native, expectedVersion, expectedVersion)
	}
	engine.api = api
	return api, nil
}

// bindLibrary opens the library at path and finds each of its functions.
func bindLibrary(path string) (api *capi, err error) {
	handle, err := openLibrary(path)
	if err != nil {
		return nil, fmt.Errorf("morflow: cannot load the engine library %s: %w", path, err)
	}
	// RegisterLibFunc panics when a function is missing.
	defer func() {
		if r := recover(); r != nil {
			api, err = nil, fmt.Errorf("morflow: %s is not a Morflow engine library: %v", path, r)
		}
	}()
	api = &capi{}
	for name, fn := range map[string]any{
		"morflow_version":       &api.version,
		"morflow_load":          &api.load,
		"morflow_from_str":      &api.fromStr,
		"morflow_pipeline_free": &api.pipelineFree,
		"morflow_param_count":   &api.paramCount,
		"morflow_param_name":    &api.paramName,
		"morflow_string_free":   &api.stringFree,
		"morflow_run":           &api.run,
		"morflow_run_args":      &api.runArgs,
		"morflow_result_error":  &api.resultError,
		"morflow_result_count":  &api.resultCount,
		"morflow_result_name":   &api.resultName,
		"morflow_result_value":  &api.resultValue,
		"morflow_result_free":   &api.resultFree,
	} {
		purego.RegisterLibFunc(fn, handle, name)
	}
	return api, nil
}

// goString copies a NUL-terminated string from engine memory ("" for nil).
func goString(p *byte) string {
	if p == nil {
		return ""
	}
	n := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(p), n)) != 0 {
		n++
	}
	return string(unsafe.Slice(p, n))
}

// cString is s as NUL-terminated bytes for the engine.
func cString(s string) []byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return b
}
