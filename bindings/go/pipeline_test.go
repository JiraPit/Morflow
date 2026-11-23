package morflow

// Runs pipelines with the real engine and the actions `morflow prep
// testdata` prepared. Without them every test here is skipped, unless
// MORFLOW_REQUIRE_ENGINE=1 (as in CI), which turns that into a failure.

import (
	"errors"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

func requireEngine(t *testing.T) {
	t.Helper()
	if _, err := loadEngine(); err != nil {
		if os.Getenv("MORFLOW_REQUIRE_ENGINE") == "1" {
			t.Fatalf("the engine is required: %v", err)
		}
		t.Skipf("no engine library: %v", err)
	}
}

func load(t *testing.T, name string) *Pipeline {
	t.Helper()
	requireEngine(t)
	p, err := Load("testdata/" + name + ".morf")
	if err != nil {
		t.Fatalf("Load(%s): %v", name, err)
	}
	t.Cleanup(func() { p.Close() })
	return p
}

func equal[T comparable](t *testing.T, what string, got, want []T) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %v, want %v", what, got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s: got %v, want %v", what, got, want)
		}
	}
}

func TestIdentityKeepsEveryDType(t *testing.T) {
	p := load(t, "identity")
	cases := []struct {
		in    any
		dtype string
		check func(*Tensor)
	}{
		{[]float32{1, 2.5, -3}, "f32", func(o *Tensor) { equal(t, "f32", o.Float32s(), []float32{1, 2.5, -3}) }},
		{[]float64{-4, 5}, "f64", func(o *Tensor) { equal(t, "f64", o.Float64s(), []float64{-4, 5}) }},
		{[]int8{-1, 2}, "i8", func(o *Tensor) { equal(t, "i8", o.Int8s(), []int8{-1, 2}) }},
		{[]int16{-300, 300}, "i16", func(o *Tensor) { equal(t, "i16", o.Int16s(), []int16{-300, 300}) }},
		{[]int32{-70000, 7}, "i32", func(o *Tensor) { equal(t, "i32", o.Int32s(), []int32{-70000, 7}) }},
		{[]int64{-1 << 40, 9}, "i64", func(o *Tensor) { equal(t, "i64", o.Int64s(), []int64{-1 << 40, 9}) }},
		{[]int{3, -4}, "i64", func(o *Tensor) { equal(t, "int", o.Int64s(), []int64{3, -4}) }},
		{[]uint32{1 << 31, 1}, "u32", func(o *Tensor) { equal(t, "u32", o.Uint32s(), []uint32{1 << 31, 1}) }},
		{[]uint64{1 << 63, 2}, "u64", func(o *Tensor) { equal(t, "u64", o.Uint64s(), []uint64{1 << 63, 2}) }},
		{TensorInput{Data: []byte{1, 255, 7}, Shape: []int{3}, DType: "u8"}, "u8", func(o *Tensor) { equal(t, "u8", o.Uint8s(), []uint8{1, 255, 7}) }},
	}
	for _, c := range cases {
		out, err := p.Run(c.in)
		if err != nil {
			t.Fatalf("%s: %v", c.dtype, err)
		}
		if out.DType != c.dtype {
			t.Fatalf("got dtype %s, want %s", out.DType, c.dtype)
		}
		c.check(out)
	}
}

func TestShapesAndOutputsGoBackIn(t *testing.T) {
	p := load(t, "identity")
	out, err := p.Run(TensorInput{Data: []float32{1, 2, 3, 4, 5, 6}, Shape: []int{2, 3}})
	if err != nil {
		t.Fatal(err)
	}
	equal(t, "shape", out.Shape, []int{2, 3})
	if out.Len() != 6 {
		t.Fatalf("Len() = %d", out.Len())
	}
	again, err := p.Run(out)
	if err != nil {
		t.Fatal(err)
	}
	equal(t, "round trip", again.Float32s(), []float32{1, 2, 3, 4, 5, 6})
	// A pointer works as well as a value.
	ptr, err := p.Run(&TensorInput{Data: []float64{8}, Shape: []int{1}})
	if err != nil {
		t.Fatal(err)
	}
	equal(t, "pointer", ptr.Float64s(), []float64{8})
}

func TestParamsAndFromStr(t *testing.T) {
	p := load(t, "resample")
	equal(t, "params", p.Params(), []string{"wav", "rate", "gain"})
	// Params returns a copy.
	p.Params()[0] = "changed"
	equal(t, "params", p.Params(), []string{"wav", "rate", "gain"})

	src, err := FromStr("from base/latest import identity\naccept Tensor $x\n$x >> identity >> emit")
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	equal(t, "params", src.Params(), []string{"x"})
}

func TestExactBundledVersions(t *testing.T) {
	p := load(t, "identity_exact")
	out, err := p.Run([]float32{4})
	if err != nil {
		t.Fatal(err)
	}
	equal(t, "exact", out.Float32s(), []float32{4})

	_, err = FromStr("from base/0.0.1 import identity\naccept Tensor $t\n$t >> identity >> emit")
	var engineErr *Error
	if !errors.As(err, &engineErr) {
		t.Fatalf("an unprepared version should fail with *Error, got %v", err)
	}
}

func TestAudioToWav(t *testing.T) {
	p := load(t, "boost")
	samples := make([]float32, 2*2205)
	for i := range samples {
		samples[i] = 0.25 * float32(math.Sin(float64(i)/10))
	}
	out, err := p.Run(Audio(samples, 2, 22050))
	if err != nil {
		t.Fatal(err)
	}
	wav := out.Bytes()
	if string(wav[0:4]) != "RIFF" || string(wav[8:12]) != "WAVE" {
		t.Fatalf("not a WAV file: % x", wav[:12])
	}
	if channels := int(wav[22]) | int(wav[23])<<8; channels != 2 {
		t.Fatalf("got %d channels", channels)
	}
	// 16-bit samples after a 44-byte header.
	if want := 44 + 2205*2*2; len(wav) != want {
		t.Fatalf("got %d bytes, want %d", len(wav), want)
	}
}

func TestBytesAndArguments(t *testing.T) {
	boost := load(t, "boost")
	samples := make([]float32, 4410)
	for i := range samples {
		samples[i] = 0.1 * float32(math.Sin(float64(i)/5))
	}
	wav, err := boost.Run(Audio(samples, 1, 44100))
	if err != nil {
		t.Fatal(err)
	}

	resample := load(t, "resample")
	rate := func(w []byte) int { return int(w[24]) | int(w[25])<<8 | int(w[26])<<16 | int(w[27])<<24 }

	// Raw bytes in, defaults for the rest.
	out, err := resample.Run(wav.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if rate(out.Bytes()) != 16000 {
		t.Fatalf("default rate: got %d", rate(out.Bytes()))
	}
	// Arguments in order, with Go ints and floats of any size.
	for _, args := range [][]any{
		{wav.Bytes(), 22050, 0.5},
		{wav.Bytes(), int64(22050), float32(0.5)},
		{wav.Bytes(), uint16(22050), 0.5},
		{wav.Bytes(), int32(22050)},
	} {
		out, err := resample.RunArgs(args...)
		if err != nil {
			t.Fatalf("%v: %v", args[1:], err)
		}
		if rate(out.Bytes()) != 22050 {
			t.Fatalf("%v: got rate %d", args[1:], rate(out.Bytes()))
		}
	}
	if _, err := resample.RunArgs(wav.Bytes(), math.NaN()); err == nil || !strings.Contains(err.Error(), "finite") {
		t.Fatalf("NaN argument: %v", err)
	}
	if _, err := resample.RunArgs(wav.Bytes(), struct{}{}); err == nil || !strings.Contains(err.Error(), "argument 1") {
		t.Fatalf("bad argument: %v", err)
	}
}

func TestStringAndBoolArguments(t *testing.T) {
	p := load(t, "stats")
	data := TensorInput{Data: []float32{3, -4}, Shape: []int{2}}
	outs, err := p.RunToMapArgs(data, false, "1")
	if err != nil {
		t.Fatal(err)
	}
	// Population std of {3, -4} is 3.5; the 1-norm is 7.
	if got := outs["spread"].Float32s()[0]; math.Abs(float64(got)-3.5) > 1e-6 {
		t.Fatalf("spread = %v", got)
	}
	if got := outs["size"].Float32s()[0]; math.Abs(float64(got)-7) > 1e-6 {
		t.Fatalf("size = %v", got)
	}
	// Defaults: unbiased std and the max norm.
	outs, err = p.RunToMapArgs(data)
	if err != nil {
		t.Fatal(err)
	}
	if got := outs["spread"].Float32s()[0]; math.Abs(float64(got)-4.9497476) > 1e-5 {
		t.Fatalf("default spread = %v", got)
	}
	if got := outs["size"].Float32s()[0]; got != 4 {
		t.Fatalf("default size = %v", got)
	}
}

func TestImages(t *testing.T) {
	p := load(t, "image")
	out, err := p.Run(Image(make([]float32, 64), 8, 8, 1))
	if err != nil {
		t.Fatal(err)
	}
	equal(t, "shape", out.Shape, []int{8, 8})
	rgb, err := p.Run(TensorInput{Data: make([]byte, 4*5*3), Shape: []int{4, 5, 3}, DType: "u8", PayloadType: "image", ColorSpace: "rgb"})
	if err != nil {
		t.Fatal(err)
	}
	equal(t, "rgb shape", rgb.Shape, []int{4, 5, 3})
}

func TestCompositesAndNamedOutputs(t *testing.T) {
	qr := load(t, "qr")
	data := []float32{1, 0, 0, 1, 1, 1}
	outs, err := qr.RunToMap(TensorInput{Data: data, Shape: []int{3, 2}})
	if err != nil {
		t.Fatal(err)
	}
	equal(t, "q", outs["q"].Shape, []int{3, 2})
	equal(t, "r", outs["r"].Shape, []int{2, 2})
	for i, v := range outs["reconstructed"].Float32s() {
		if math.Abs(float64(v-data[i])) > 1e-5 {
			t.Fatalf("reconstructed[%d] = %v, want %v", i, v, data[i])
		}
	}

	add := load(t, "add")
	sum, err := add.Run([]any{TensorInput{Data: []int32{1, 2, 3, 4, 5, 6}, Shape: []int{2, 3}}, []int32{10, 20, 30}})
	if err != nil {
		t.Fatal(err)
	}
	equal(t, "add", sum.Int32s(), []int32{11, 22, 33, 14, 25, 36})

	pass, err := FromStr("accept Composite $pair\n$pair >> emit")
	if err != nil {
		t.Fatal(err)
	}
	defer pass.Close()
	pair, err := pass.Run([]any{[]float64{1.5}, []int64{7}})
	if err != nil {
		t.Fatal(err)
	}
	if !pair.IsComposite() || len(pair.Items) != 2 || pair.String() != "Tensor(composite of 2)" {
		t.Fatalf("got %v", pair)
	}
	equal(t, "first", pair.Items[0].Float64s(), []float64{1.5})
	equal(t, "second", pair.Items[1].Int64s(), []int64{7})
	// A Composite output goes back in as a Composite.
	again, err := pass.Run(pair)
	if err != nil {
		t.Fatal(err)
	}
	equal(t, "again", again.Items[1].Int64s(), []int64{7})
}

func TestScalarsReshapeAndSums(t *testing.T) {
	relu := load(t, "relu")
	out, err := relu.RunArgs(-3.0)
	if err != nil {
		t.Fatal(err)
	}
	equal(t, "relu", out.Float32s(), []float32{0})

	reshape := load(t, "reshape")
	shaped, err := reshape.Run(TensorInput{Data: make([]float32, 2000), Shape: []int{4, 500}})
	if err != nil {
		t.Fatal(err)
	}
	equal(t, "reshape", shaped.Shape, []int{2, 1000})

	sum := load(t, "sum")
	total, err := sum.Run(TensorInput{Data: []int32{1, 2, 3, 4}, Shape: []int{2, 2}})
	if err != nil {
		t.Fatal(err)
	}
	if total.DType != "i64" {
		t.Fatalf("sum dtype %s", total.DType)
	}
	equal(t, "sum", total.Int64s(), []int64{4, 6})
	values, err := total.Values()
	if err != nil {
		t.Fatal(err)
	}
	equal(t, "values", values.([]int64), []int64{4, 6})
}

func TestEngineErrors(t *testing.T) {
	requireEngine(t)
	var engineErr *Error
	_, err := FromStr("accept Tensor $t\n$t >> no_such_action >> emit")
	if !errors.As(err, &engineErr) || !strings.Contains(engineErr.Message, "no_such_action") {
		t.Fatalf("unknown action: %v", err)
	}
	_, err = FromStr("this is not a pipeline")
	if !errors.As(err, &engineErr) {
		t.Fatalf("parse error: %v", err)
	}
	_, err = Load("testdata/does-not-exist.morf")
	if !errors.As(err, &engineErr) {
		t.Fatalf("missing file: %v", err)
	}
	_, err = FromStr("import linalg_blas/latest\naccept Tensor $t\n$t >> matmul >> emit")
	if !errors.As(err, &engineErr) || !strings.Contains(engineErr.Message, "No prepared catalog for linalg_blas/latest") {
		t.Fatalf("unprepared pack: %v", err)
	}

	p := load(t, "qr")
	if _, err := p.Run(TensorInput{Data: []float32{1, 2}, Shape: []int{2}}); !errors.As(err, &engineErr) {
		t.Fatalf("wrong shape: %v", err)
	}
	if _, err := p.Run(nil); !errors.As(err, &engineErr) {
		t.Fatalf("missing input: %v", err)
	}
	if _, err := p.Run(map[string]int{}); err == nil || !strings.Contains(err.Error(), "unsupported pipeline input map[string]int") {
		t.Fatalf("unsupported input: %v", err)
	}
}

func TestCloseAndConcurrency(t *testing.T) {
	p := load(t, "identity")
	var wg sync.WaitGroup
	failures := make(chan error, 50)
	for i := range 50 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out, err := p.Run([]float32{float32(i)})
			if err != nil {
				failures <- err
				return
			}
			if got := out.Float32s()[0]; got != float32(i) {
				failures <- errors.New("a run returned another run's output")
			}
		}()
	}
	wg.Wait()
	close(failures)
	for err := range failures {
		t.Fatal(err)
	}

	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
	if err := p.Close(); err != nil {
		t.Fatal("closing twice:", err)
	}
	if _, err := p.Run([]float32{1}); !errors.Is(err, ErrClosed) {
		t.Fatalf("run after Close: %v", err)
	}
	if _, err := p.RunToMapArgs(); !errors.Is(err, ErrClosed) {
		t.Fatalf("RunToMapArgs after Close: %v", err)
	}
}

func TestCloseWaitsForRunsInProgress(t *testing.T) {
	p := load(t, "reshape")
	input := TensorInput{Data: make([]float32, 2000), Shape: []int{4, 500}}
	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 200 {
				if _, err := p.Run(input); err != nil && !errors.Is(err, ErrClosed) {
					errs <- err
					return
				}
			}
		}()
	}
	time.Sleep(5 * time.Millisecond)
	p.Close()
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.running != 0 || p.handle != nil {
		t.Fatalf("after Close and the runs: running=%d, handle freed=%v", p.running, p.handle == nil)
	}
}

func TestUnclosedPipelinesAreFreedByTheGarbageCollector(t *testing.T) {
	requireEngine(t)
	for range 20 {
		p, err := Load("testdata/identity.morf")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := p.Run([]float32{1}); err != nil {
			t.Fatal(err)
		}
	}
	// The cleanups run on their own goroutine; this only checks they don't
	// crash the process.
	runtime.GC()
	runtime.GC()
	time.Sleep(50 * time.Millisecond)
}
