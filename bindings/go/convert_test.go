package morflow

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"unsafe"
)

func TestInputs(t *testing.T) {
	for _, c := range []struct {
		in    any
		kind  uint32
		dtype string
		shape []int
		size  int
	}{
		{[]byte{1, 2, 3}, kindBytes, "", nil, 3},
		{[]float32{1, 2}, kindTensor, "f32", []int{2}, 8},
		{[]float64{1}, kindTensor, "f64", []int{1}, 8},
		{[]int8{1, 2, 3}, kindTensor, "i8", []int{3}, 3},
		{[]int16{1}, kindTensor, "i16", []int{1}, 2},
		{[]int32{1}, kindTensor, "i32", []int{1}, 4},
		{[]int64{1}, kindTensor, "i64", []int{1}, 8},
		{[]int{1, 2}, kindTensor, "i64", []int{2}, 16},
		{[]uint32{1}, kindTensor, "u32", []int{1}, 4},
		{[]uint64{1}, kindTensor, "u64", []int{1}, 8},
		{[]float32{}, kindTensor, "f32", []int{0}, 0},
		{TensorInput{Data: []float32{1, 2, 3, 4}, Shape: []int{2, 2}}, kindTensor, "f32", []int{2, 2}, 16},
		{TensorInput{Data: []byte{0, 0, 128, 63}, Shape: []int{1}}, kindTensor, "", []int{1}, 4},
		{TensorInput{Data: []int32{1}, Shape: []int{1}, DType: "f32"}, kindTensor, "f32", []int{1}, 4},
		{&TensorInput{Data: []int16{1}, Shape: []int{1}}, kindTensor, "i16", []int{1}, 2},
		{&Tensor{Shape: []int{1}, DType: "u8", Data: []byte{9}}, kindTensor, "u8", []int{1}, 1},
	} {
		v, err := input(c.in)
		if err != nil {
			t.Fatalf("%T: %v", c.in, err)
		}
		if v.kind != c.kind || v.dtype != c.dtype || len(v.data) != c.size || len(v.shape) != len(c.shape) {
			t.Fatalf("%T: got kind %d dtype %q shape %v %d bytes", c.in, v.kind, v.dtype, v.shape, len(v.data))
		}
		for i := range c.shape {
			if v.shape[i] != c.shape[i] {
				t.Fatalf("%T: shape %v", c.in, v.shape)
			}
		}
	}
}

func TestInputsShareMemory(t *testing.T) {
	data := []float32{1}
	v, _ := input(data)
	if unsafe.Pointer(&v.data[0]) != unsafe.Pointer(&data[0]) {
		t.Fatal("a numeric slice should be sent without a copy")
	}
}

func TestAudioAndImageInputs(t *testing.T) {
	a := Audio(make([]float32, 300), 3, 8000)
	if a.PayloadType != "audio" || a.SampleRate != 8000 || a.Channels != 3 || a.Shape[0] != 3 || a.Shape[1] != 100 {
		t.Fatalf("Audio: %+v", a)
	}
	if empty := Audio(nil, 0, 8000); empty.Shape[1] != 0 {
		t.Fatalf("Audio with no channels: %+v", empty)
	}
	img := Image(make([]byte, 12), 2, 2, 3)
	if img.PayloadType != "image" || img.Shape[2] != 3 {
		t.Fatalf("Image: %+v", img)
	}
}

func TestInputErrors(t *testing.T) {
	for _, c := range []struct {
		in      any
		message string
	}{
		{nil, "cannot be nil"},
		{(*TensorInput)(nil), "nil *TensorInput"},
		{(*Tensor)(nil), "nil *Tensor"},
		{"text", "unsupported pipeline input string"},
		{[]string{"a"}, "unsupported pipeline input []string"},
		{TensorInput{Data: []float32{1}}, "Shape is required"},
		{TensorInput{Data: []float32{1}, Shape: []int{-1}}, "negative dimension"},
		{TensorInput{Data: []string{"a"}, Shape: []int{1}}, "must be a numeric slice or a []byte"},
		{TensorInput{Data: []float32{1}, Shape: []int{1}, SampleRate: -1}, "must not be negative"},
		{[]any{[]byte{1}}, "Composite part 0 is []uint8"},
		{[]any{[]float32{1}, nil}, "Composite part 1"},
		{&Tensor{DType: "composite", Items: []*Tensor{nil}}, "Composite part 0"},
	} {
		if _, err := input(c.in); err == nil || !strings.Contains(err.Error(), c.message) {
			t.Fatalf("%#v: got %v, want %q", c.in, err, c.message)
		}
	}
}

func TestComposites(t *testing.T) {
	v, err := input([]any{[]float32{1}, []any{[]int32{2}, []int64{3}}})
	if err != nil {
		t.Fatal(err)
	}
	if v.payloadType != "composite" || len(v.items) != 2 || len(v.items[1].items) != 2 {
		t.Fatalf("got %+v", v)
	}
	back, err := input(&Tensor{DType: "composite", Items: []*Tensor{{DType: "f32", Shape: []int{1}, Data: make([]byte, 4)}}})
	if err != nil || len(back.items) != 1 || back.items[0].dtype != "f32" {
		t.Fatalf("got %+v, %v", back, err)
	}
}

func TestArguments(t *testing.T) {
	numbers := []any{int(1), int8(1), int16(1), int32(1), int64(1), uint(1), uint8(1), uint16(1), uint32(1), uint64(1), float32(1), float64(1)}
	for _, n := range numbers {
		v, err := argument(n)
		if err != nil || v.kind != kindNumber || v.number != 1 {
			t.Fatalf("%T: %+v, %v", n, v, err)
		}
	}
	if v, _ := argument("hi"); v.kind != kindText || string(v.data) != "hi" {
		t.Fatalf("string: %+v", v)
	}
	if v, _ := argument(true); v.kind != kindBool || !v.flag {
		t.Fatalf("bool: %+v", v)
	}
	if v, _ := argument([]float32{1}); v.kind != kindTensor {
		t.Fatalf("slice: %+v", v)
	}
	if _, err := argument(struct{}{}); err == nil {
		t.Fatal("a struct is not an argument")
	}
}

func TestEncoding(t *testing.T) {
	var e encoder
	defer e.release()

	if in, err := e.encodeInput(nil, false); in != nil || err != nil {
		t.Fatal("no input should encode as nil")
	}
	in, err := e.encodeInput(TensorInput{Data: []float32{1, 2}, Shape: []int{1, 2}, PayloadType: "audio", SampleRate: 8000, Channels: 1, Layout: "planar", ColorSpace: "rgb", DType: "f32"}, true)
	if err != nil {
		t.Fatal(err)
	}
	if in.kind != kindTensor || in.rank != 2 || in.dataLen != 8 || in.sampleRate != 8000 || in.channels != 1 {
		t.Fatalf("got %+v", *in)
	}
	for _, s := range []struct {
		p    unsafe.Pointer
		want string
	}{{in.dtype, "f32"}, {in.payloadType, "audio"}, {in.layout, "planar"}, {in.colorSpace, "rgb"}} {
		if got := goString((*byte)(s.p)); got != s.want {
			t.Fatalf("got %q, want %q", got, s.want)
		}
	}
	shape := unsafe.Slice((*uint64)(in.shape), 2)
	if shape[0] != 1 || shape[1] != 2 {
		t.Fatalf("shape %v", shape)
	}

	plain, _ := e.encodeInput([]float32{1}, true)
	if plain.sampleRate != -1 || plain.channels != -1 || plain.payloadType != nil {
		t.Fatalf("absent fields should be -1 and nil: %+v", *plain)
	}
	if _, err := e.encodeInput("bad", true); err == nil {
		t.Fatal("a bad input should fail")
	}

	args, err := e.encodeArgs([]any{[]byte{7}, 2.5, "x", true, false, []any{[]float32{1}, []float32{2}}})
	if err != nil {
		t.Fatal(err)
	}
	list := unsafe.Slice(args, 6)
	if list[0].kind != kindBytes || list[0].dataLen != 1 || list[1].number != 2.5 || list[2].kind != kindText || list[3].flag != 1 || list[4].flag != 0 || list[5].itemCount != 2 {
		t.Fatalf("got %+v", list)
	}
	if none, err := e.encodeArgs(nil); none != nil || err != nil {
		t.Fatal("no arguments should encode as nil")
	}
	for _, bad := range []any{math.Inf(1), struct{}{}, []any{[]byte{1}}} {
		if _, err := e.encodeArgs([]any{bad}); err == nil || !strings.Contains(err.Error(), "argument 0") {
			t.Fatalf("%v: %v", bad, err)
		}
	}
}

// out builds a MorflowValueOut in Go memory, as the engine lays it out.
func out(dtype string, shape []uint64, data []byte, items []valueOut) valueOut {
	v := valueOut{dtype: &cString(dtype)[0], rank: uintptr(len(shape)), dataLen: uintptr(len(data)), itemCount: uintptr(len(items))}
	if len(shape) > 0 {
		v.shape = unsafe.Pointer(&shape[0])
	}
	if len(data) > 0 {
		v.data = unsafe.Pointer(&data[0])
	}
	if len(items) > 0 {
		v.items = unsafe.Pointer(&items[0])
	}
	return v
}

func TestReadingOutputs(t *testing.T) {
	data := []byte{0, 0, 128, 63, 0, 0, 0, 64}
	tensor := out("f32", []uint64{2, 1}, data, nil)
	got := readValue(&tensor)
	if got.DType != "f32" || got.Shape[0] != 2 || got.Shape[1] != 1 || got.Float32s()[1] != 2 {
		t.Fatalf("got %v %v", got, got.Float32s())
	}
	data[0] = 9
	if got.Data[0] == 9 {
		t.Fatal("outputs must be copied out of engine memory")
	}
	scalar := out("f64", nil, make([]byte, 8), nil)
	if s := readValue(&scalar); len(s.Shape) != 0 || s.Len() != 1 {
		t.Fatalf("scalar: %v", s)
	}
	parts := []valueOut{out("i32", []uint64{1}, []byte{5, 0, 0, 0}, nil), out("u8", []uint64{0}, nil, nil)}
	pair := out("composite", nil, nil, parts)
	c := readValue(&pair)
	if !c.IsComposite() || len(c.Items) != 2 || c.Items[0].Int32s()[0] != 5 || len(c.Items[1].Data) != 0 {
		t.Fatalf("composite: %v", c)
	}
}

// stub is a C API whose results are Go values, for decodeResult.
func stub(errMessage string, names []string, values []valueOut) (*capi, *int) {
	freed := 0
	api := &capi{
		resultError: func(unsafe.Pointer) *byte {
			if errMessage == "" {
				return nil
			}
			return &cString(errMessage)[0]
		},
		resultCount: func(unsafe.Pointer) uintptr { return uintptr(len(values)) },
		resultName:  func(_ unsafe.Pointer, i uintptr) *byte { return &cString(names[i])[0] },
		resultValue: func(_ unsafe.Pointer, i uintptr) *valueOut { return &values[i] },
		resultFree:  func(unsafe.Pointer) { freed++ },
	}
	return api, &freed
}

func TestDecodingResults(t *testing.T) {
	handle := unsafe.Pointer(new(byte))
	one := []valueOut{out("i32", []uint64{1}, []byte{3, 0, 0, 0}, nil), out("i32", []uint64{1}, []byte{4, 0, 0, 0}, nil)}

	api, freed := stub("", []string{"a", "b"}, one)
	single, _, err := decodeResult(api, handle, false)
	if err != nil || single.Int32s()[0] != 3 || *freed != 1 {
		t.Fatalf("single: %v, %v, freed %d", single, err, *freed)
	}
	_, named, err := decodeResult(api, handle, true)
	if err != nil || len(named) != 2 || named["b"].Int32s()[0] != 4 || *freed != 2 {
		t.Fatalf("named: %v, %v", named, err)
	}

	api, freed = stub("Execution error: boom", nil, nil)
	_, _, err = decodeResult(api, handle, false)
	if e, ok := err.(*Error); !ok || e.Message != "Execution error: boom" || e.Error() != e.Message || *freed != 1 {
		t.Fatalf("error result: %v, freed %d", err, *freed)
	}
	api, _ = stub("", nil, nil)
	if _, _, err := decodeResult(api, handle, false); err == nil || !strings.Contains(err.Error(), "emitted nothing") {
		t.Fatalf("empty result: %v", err)
	}
	if _, _, err := decodeResult(api, nil, false); err == nil {
		t.Fatal("a nil result should fail")
	}
}

func TestTensorAccessors(t *testing.T) {
	cases := map[string]struct {
		data []byte
		want any
	}{
		"f32": {[]byte{0, 0, 128, 63}, []float32{1}},
		"f64": {[]byte{0, 0, 0, 0, 0, 0, 240, 63}, []float64{1}},
		"u8":  {[]byte{7}, []uint8{7}},
		"i8":  {[]byte{255}, []int8{-1}},
		"i16": {[]byte{255, 255}, []int16{-1}},
		"i32": {[]byte{1, 0, 0, 0}, []int32{1}},
		"i64": {[]byte{1, 0, 0, 0, 0, 0, 0, 0}, []int64{1}},
		"u32": {[]byte{1, 0, 0, 0}, []uint32{1}},
		"u64": {[]byte{1, 0, 0, 0, 0, 0, 0, 0}, []uint64{1}},
	}
	for dtype, c := range cases {
		tensor := &Tensor{DType: dtype, Shape: []int{1}, Data: c.data}
		values, err := tensor.Values()
		if err != nil {
			t.Fatal(err)
		}
		if got, want := fmt.Sprint(values), fmt.Sprint(c.want); got != want {
			t.Fatalf("%s: got %s, want %s", dtype, got, want)
		}
	}
	if _, err := (&Tensor{DType: "composite"}).Values(); err == nil {
		t.Fatal("a composite has no element slice")
	}
	u8 := &Tensor{DType: "u8", Data: []byte{1}}
	u8.Uint8s()[0] = 9
	if u8.Data[0] != 1 || u8.Bytes()[0] != 1 {
		t.Fatal("Uint8s must copy")
	}
	if s := (&Tensor{DType: "f32", Shape: []int{2, 3}}).String(); s != "Tensor(f32 [2 3])" {
		t.Fatal(s)
	}
	if len((&Tensor{}).Float32s()) != 0 {
		t.Fatal("no data, no elements")
	}
}

func TestCStrings(t *testing.T) {
	if goString(nil) != "" || goString(&cString("héllo")[0]) != "héllo" {
		t.Fatal("goString")
	}
	if b := cString("ab"); len(b) != 3 || b[2] != 0 {
		t.Fatal("cString")
	}
}

func TestBindingBadLibraries(t *testing.T) {
	if _, err := bindLibrary(filepath.Join(t.TempDir(), "missing.so")); err == nil || !strings.Contains(err.Error(), "cannot load the engine library") {
		t.Fatalf("missing library: %v", err)
	}
	notALibrary := filepath.Join(t.TempDir(), "libnot.so")
	os.WriteFile(notALibrary, []byte("text"), 0o644)
	if _, err := bindLibrary(notALibrary); err == nil {
		t.Fatal("a text file is not a library")
	}
	var libc string
	switch runtime.GOOS {
	case "linux":
		for _, p := range []string{"/lib/x86_64-linux-gnu/libc.so.6", "/lib64/libc.so.6", "/lib/aarch64-linux-gnu/libc.so.6", "/usr/lib64/libc.so.6"} {
			if _, err := os.Stat(p); err == nil {
				libc = p
				break
			}
		}
	case "darwin":
		libc = "/usr/lib/libSystem.B.dylib"
	case "windows":
		libc = `C:\Windows\System32\kernel32.dll`
	}
	if libc == "" {
		t.Skip("no system library to try")
	}
	if _, err := bindLibrary(libc); err == nil || !strings.Contains(err.Error(), "is not a Morflow engine library") {
		t.Fatalf("a library without Morflow's functions: %v", err)
	}
}

func TestLoadingFailuresAreNotRemembered(t *testing.T) {
	engine.Lock()
	saved := engine.api
	engine.api = nil
	engine.Unlock()
	defer func() {
		engine.Lock()
		engine.api = saved
		engine.Unlock()
	}()

	t.Setenv("MORFLOW_CAPI_PATH", filepath.Join(t.TempDir(), "missing.so"))
	if _, err := FromStr("accept Tensor $t\n$t >> emit"); err == nil || !strings.Contains(err.Error(), "MORFLOW_CAPI_PATH") {
		t.Fatalf("FromStr: %v", err)
	}
	notALibrary := filepath.Join(t.TempDir(), "libnot.so")
	os.WriteFile(notALibrary, []byte("text"), 0o644)
	t.Setenv("MORFLOW_CAPI_PATH", notALibrary)
	if _, err := Load("testdata/identity.morf"); err == nil || !strings.Contains(err.Error(), "cannot load") {
		t.Fatalf("Load: %v", err)
	}
}

func TestEngineVersionMismatch(t *testing.T) {
	requireEngine(t)
	engine.Lock()
	saved := engine.api
	engine.api = nil
	engine.Unlock()
	expectedVersion = "9.9.9"
	defer func() {
		expectedVersion = EngineVersion
		engine.Lock()
		engine.api = saved
		engine.Unlock()
	}()
	if _, err := FromStr("accept Tensor $t\n$t >> emit"); err == nil || !strings.Contains(err.Error(), "but this package runs engine 9.9.9") {
		t.Fatalf("got %v", err)
	}
}
