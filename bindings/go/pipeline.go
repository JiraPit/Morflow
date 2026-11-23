package morflow

import (
	"runtime"
	"sync"
	"unsafe"
)

// Pipeline is a compiled pipeline. Run it from as many goroutines as you
// like: each run works on its own copy inside the engine.
//
// A pipeline holds engine memory. Close releases it once runs in progress
// finish; a pipeline that is never closed releases it when it is garbage
// collected.
type Pipeline struct {
	api     *capi
	params  []string
	mu      sync.Mutex
	handle  unsafe.Pointer
	running int
	closed  bool
	cleanup runtime.Cleanup
}

// FromStr compiles pipeline source text.
func FromStr(source string) (*Pipeline, error) {
	return compile(source, func(api *capi) func(*byte, *unsafe.Pointer) unsafe.Pointer { return api.fromStr })
}

// Load compiles the .morf pipeline file at path.
func Load(path string) (*Pipeline, error) {
	return compile(path, func(api *capi) func(*byte, *unsafe.Pointer) unsafe.Pointer { return api.load })
}

func compile(text string, pick func(*capi) func(*byte, *unsafe.Pointer) unsafe.Pointer) (*Pipeline, error) {
	api, err := loadEngine()
	if err != nil {
		return nil, err
	}
	var pinner runtime.Pinner
	defer pinner.Unpin()
	source := cString(text)
	pinner.Pin(&source[0])
	errOut := new(unsafe.Pointer)
	pinner.Pin(errOut)
	handle := pick(api)(&source[0], errOut)
	if handle == nil {
		message := "the pipeline could not be compiled"
		if *errOut != nil {
			message = goString((*byte)(*errOut))
			api.stringFree((*byte)(*errOut))
		}
		return nil, &Error{Message: message}
	}
	p := &Pipeline{api: api, handle: handle}
	for i := uintptr(0); i < api.paramCount(handle); i++ {
		name := api.paramName(handle, i)
		p.params = append(p.params, goString(name))
		api.stringFree(name)
	}
	free := api.pipelineFree
	p.cleanup = runtime.AddCleanup(p, func(h unsafe.Pointer) { free(h) }, handle)
	return p, nil
}

// Params lists the parameters declared with accept $name, in order.
func (p *Pipeline) Params() []string { return append([]string(nil), p.params...) }

// Run runs the pipeline on input and returns its output. A pipeline that
// declares no input takes nil.
//
// The input is a TensorInput, a numeric slice ([]float32, []int32, …) for a
// one-dimensional tensor, a []byte of raw bytes such as a WAV or PNG file, a
// *Tensor from an earlier run, or a []any of those for a Composite.
func (p *Pipeline) Run(input any) (*Tensor, error) {
	out, _, err := p.run(input, false, nil, false)
	return out, err
}

// RunToMap runs the pipeline on input and returns every output named with
// emit("name").
func (p *Pipeline) RunToMap(input any) (map[string]*Tensor, error) {
	_, outs, err := p.run(input, false, nil, true)
	return outs, err
}

// RunArgs runs the pipeline with its parameters in declaration order.
// Parameters left out use their declared defaults. Each argument is anything
// Run accepts, a number (Scalar or a number argument), a string (StrArg) or
// a bool (BoolArg).
func (p *Pipeline) RunArgs(args ...any) (*Tensor, error) {
	out, _, err := p.run(nil, true, args, false)
	return out, err
}

// RunToMapArgs runs the pipeline with its parameters in declaration order
// and returns every named output.
func (p *Pipeline) RunToMapArgs(args ...any) (map[string]*Tensor, error) {
	_, outs, err := p.run(nil, true, args, true)
	return outs, err
}

// Close releases the pipeline's engine memory once runs in progress finish.
// Later runs return ErrClosed. Closing twice does nothing.
func (p *Pipeline) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil
	}
	p.closed = true
	p.cleanup.Stop()
	if p.running == 0 {
		p.free()
	}
	return nil
}

func (p *Pipeline) free() {
	if p.handle != nil {
		p.api.pipelineFree(p.handle)
		p.handle = nil
	}
}

func (p *Pipeline) run(in any, withArgs bool, args []any, all bool) (*Tensor, map[string]*Tensor, error) {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil, nil, ErrClosed
	}
	p.running++
	handle := p.handle
	p.mu.Unlock()
	defer func() {
		p.mu.Lock()
		p.running--
		if p.closed && p.running == 0 {
			p.free()
		}
		p.mu.Unlock()
		// The pipeline must stay reachable until the engine call returns,
		// or its cleanup could free the handle mid-run.
		runtime.KeepAlive(p)
	}()

	var e encoder
	defer e.release()
	flag := uint8(0)
	if all {
		flag = 1
	}
	var result unsafe.Pointer
	if withArgs {
		values, err := e.encodeArgs(args)
		if err != nil {
			return nil, nil, err
		}
		result = p.api.runArgs(handle, values, uintptr(len(args)), flag)
	} else {
		value, err := e.encodeInput(in, in != nil)
		if err != nil {
			return nil, nil, err
		}
		result = p.api.run(handle, value, flag)
	}
	return decodeResult(p.api, result, all)
}
