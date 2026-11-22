package morflow

// Converts Go values to engine inputs and engine results to Go values, with
// the same rules as the Node.js, Python and Dart packages.

import (
	"fmt"
	"math"
	"runtime"
	"unsafe"
)

// value is an input normalized to one of the engine's value kinds.
type value struct {
	kind        uint32
	data        []byte
	shape       []int
	dtype       string
	payloadType string
	colorSpace  string
	layout      string
	sampleRate  int
	channels    int
	items       []value
	number      float64
	flag        bool
}

// sliceTensor is a numeric slice as a one-dimensional tensor of its dtype,
// sharing its memory. ok is false for anything that is not a numeric slice.
func sliceTensor(data any) (raw []byte, dtype string, n int, ok bool) {
	switch s := data.(type) {
	case []float32:
		return bytesOf(s), "f32", len(s), true
	case []float64:
		return bytesOf(s), "f64", len(s), true
	case []int8:
		return bytesOf(s), "i8", len(s), true
	case []int16:
		return bytesOf(s), "i16", len(s), true
	case []int32:
		return bytesOf(s), "i32", len(s), true
	case []int64:
		return bytesOf(s), "i64", len(s), true
	case []uint32:
		return bytesOf(s), "u32", len(s), true
	case []uint64:
		return bytesOf(s), "u64", len(s), true
	case []int:
		// Go's int has no fixed size; it is sent as i64.
		wide := make([]int64, len(s))
		for i, v := range s {
			wide[i] = int64(v)
		}
		return bytesOf(wide), "i64", len(s), true
	}
	return nil, "", 0, false
}

const inputTypes = "a TensorInput, a numeric slice ([]float32, []float64, []int8, []int16, []int32, []int64, []int, []uint32, []uint64), a []byte of raw bytes, a *Tensor from an earlier run, or a []any of those (a Composite)"

// input normalizes a pipeline input.
func input(x any) (value, error) {
	switch v := x.(type) {
	case nil:
		return value{}, fmt.Errorf("morflow: a pipeline input cannot be nil; pass %s", inputTypes)
	case []byte:
		return value{kind: kindBytes, data: v}, nil
	case TensorInput:
		return tensorInput(v)
	case *TensorInput:
		if v == nil {
			return value{}, fmt.Errorf("morflow: a pipeline input cannot be a nil *TensorInput")
		}
		return tensorInput(*v)
	case *Tensor:
		if v == nil {
			return value{}, fmt.Errorf("morflow: a pipeline input cannot be a nil *Tensor")
		}
		return fromTensor(v)
	case []any:
		return composite(v)
	}
	if raw, dtype, n, ok := sliceTensor(x); ok {
		return value{kind: kindTensor, data: raw, shape: []int{n}, dtype: dtype}, nil
	}
	return value{}, fmt.Errorf("morflow: unsupported pipeline input %T; pass %s", x, inputTypes)
}

func tensorInput(t TensorInput) (value, error) {
	v := value{
		kind:        kindTensor,
		shape:       t.Shape,
		dtype:       t.DType,
		payloadType: t.PayloadType,
		colorSpace:  t.ColorSpace,
		layout:      t.Layout,
		sampleRate:  t.SampleRate,
		channels:    t.Channels,
	}
	if t.Shape == nil {
		return value{}, fmt.Errorf("morflow: TensorInput.Shape is required")
	}
	for _, d := range t.Shape {
		if d < 0 {
			return value{}, fmt.Errorf("morflow: TensorInput.Shape %v has a negative dimension", t.Shape)
		}
	}
	if t.SampleRate < 0 || t.Channels < 0 {
		return value{}, fmt.Errorf("morflow: TensorInput.SampleRate and Channels must not be negative")
	}
	switch d := t.Data.(type) {
	case []byte:
		v.data = d
	default:
		raw, dtype, _, ok := sliceTensor(t.Data)
		if !ok {
			return value{}, fmt.Errorf("morflow: TensorInput.Data must be a numeric slice or a []byte, not %T", t.Data)
		}
		v.data = raw
		if v.dtype == "" {
			v.dtype = dtype
		}
	}
	return v, nil
}

func fromTensor(t *Tensor) (value, error) {
	if t.IsComposite() {
		parts := make([]any, len(t.Items))
		for i, item := range t.Items {
			parts[i] = item
		}
		return composite(parts)
	}
	return value{kind: kindTensor, data: t.Data, shape: t.Shape, dtype: t.DType}, nil
}

func composite(parts []any) (value, error) {
	v := value{kind: kindTensor, shape: []int{}, payloadType: "composite"}
	for i, part := range parts {
		item, err := input(part)
		if err != nil {
			return value{}, fmt.Errorf("morflow: Composite part %d: %w", i, err)
		}
		if item.kind != kindTensor {
			return value{}, fmt.Errorf("morflow: Composite part %d is %T; parts must be tensors (TensorInputs, numeric slices, *Tensors or []any), not raw bytes", i, part)
		}
		v.items = append(v.items, item)
	}
	return v, nil
}

// argument normalizes a positional parameter: anything input accepts, a
// number (Scalar or *Arg), a string (StrArg) or a bool (BoolArg).
func argument(x any) (value, error) {
	switch v := x.(type) {
	case bool:
		return value{kind: kindBool, flag: v}, nil
	case string:
		return value{kind: kindText, data: []byte(v)}, nil
	case float64:
		return value{kind: kindNumber, number: v}, nil
	case float32:
		return value{kind: kindNumber, number: float64(v)}, nil
	case int:
		return value{kind: kindNumber, number: float64(v)}, nil
	case int8:
		return value{kind: kindNumber, number: float64(v)}, nil
	case int16:
		return value{kind: kindNumber, number: float64(v)}, nil
	case int32:
		return value{kind: kindNumber, number: float64(v)}, nil
	case int64:
		return value{kind: kindNumber, number: float64(v)}, nil
	case uint:
		return value{kind: kindNumber, number: float64(v)}, nil
	case uint8:
		return value{kind: kindNumber, number: float64(v)}, nil
	case uint16:
		return value{kind: kindNumber, number: float64(v)}, nil
	case uint32:
		return value{kind: kindNumber, number: float64(v)}, nil
	case uint64:
		return value{kind: kindNumber, number: float64(v)}, nil
	}
	return input(x)
}

// encoder builds engine inputs from Go memory, pinning everything the
// engine reads so the garbage collector leaves it alone during the call.
type encoder struct {
	pinner runtime.Pinner
}

func (e *encoder) release() { e.pinner.Unpin() }

func (e *encoder) bytes(b []byte) unsafe.Pointer {
	if len(b) == 0 {
		return nil
	}
	e.pinner.Pin(&b[0])
	return unsafe.Pointer(&b[0])
}

func (e *encoder) text(s string) unsafe.Pointer {
	if s == "" {
		return nil
	}
	return e.bytes(cString(s))
}

func (e *encoder) write(v value, target *valueIn) error {
	*target = valueIn{kind: v.kind, sampleRate: -1, channels: -1}
	switch v.kind {
	case kindTensor:
		shape := make([]uint64, max(len(v.shape), 1))
		for i, d := range v.shape {
			shape[i] = uint64(d)
		}
		e.pinner.Pin(&shape[0])
		target.data = e.bytes(v.data)
		target.dataLen = uintptr(len(v.data))
		target.shape = unsafe.Pointer(&shape[0])
		target.rank = uintptr(len(v.shape))
		target.dtype = e.text(v.dtype)
		target.payloadType = e.text(v.payloadType)
		target.colorSpace = e.text(v.colorSpace)
		target.layout = e.text(v.layout)
		if v.sampleRate > 0 {
			target.sampleRate = int64(v.sampleRate)
		}
		if v.channels > 0 {
			target.channels = int64(v.channels)
		}
		if len(v.items) > 0 {
			items := make([]valueIn, len(v.items))
			e.pinner.Pin(&items[0])
			for i, item := range v.items {
				if err := e.write(item, &items[i]); err != nil {
					return err
				}
			}
			target.items = unsafe.Pointer(&items[0])
			target.itemCount = uintptr(len(items))
		}
	case kindBytes, kindText:
		target.data = e.bytes(v.data)
		target.dataLen = uintptr(len(v.data))
	case kindNumber:
		if math.IsNaN(v.number) || math.IsInf(v.number, 0) {
			return fmt.Errorf("morflow: a pipeline argument must be a finite number, not %v", v.number)
		}
		target.number = v.number
	case kindBool:
		if v.flag {
			target.flag = 1
		}
	}
	return nil
}

// encodeInput is the engine form of a single-input run's input, or nil for
// a pipeline run without one.
func (e *encoder) encodeInput(x any, hasInput bool) (*valueIn, error) {
	if !hasInput {
		return nil, nil
	}
	v, err := input(x)
	if err != nil {
		return nil, err
	}
	target := new(valueIn)
	e.pinner.Pin(target)
	return target, e.write(v, target)
}

// encodeArgs is the engine form of positional parameters, in order.
func (e *encoder) encodeArgs(args []any) (*valueIn, error) {
	if len(args) == 0 {
		return nil, nil
	}
	values := make([]value, len(args))
	for i, a := range args {
		v, err := argument(a)
		if err != nil {
			return nil, fmt.Errorf("morflow: argument %d: %w", i, err)
		}
		values[i] = v
	}
	list := make([]valueIn, len(values))
	e.pinner.Pin(&list[0])
	for i, v := range values {
		if err := e.write(v, &list[i]); err != nil {
			return nil, fmt.Errorf("morflow: argument %d: %w", i, err)
		}
	}
	return &list[0], nil
}

// readValue copies an emitted value out of engine memory, which is freed
// with its result.
func readValue(out *valueOut) *Tensor {
	t := &Tensor{DType: goString(out.dtype)}
	if t.IsComposite() {
		items := unsafe.Slice((*valueOut)(out.items), out.itemCount)
		t.Items = make([]*Tensor, len(items))
		for i := range items {
			t.Items[i] = readValue(&items[i])
		}
		t.Shape = []int{}
		return t
	}
	t.Shape = make([]int, out.rank)
	if out.rank > 0 {
		for i, d := range unsafe.Slice((*uint64)(out.shape), out.rank) {
			t.Shape[i] = int(d)
		}
	}
	t.Data = make([]byte, out.dataLen)
	if out.dataLen > 0 {
		copy(t.Data, unsafe.Slice((*byte)(out.data), out.dataLen))
	}
	return t
}

// decodeResult reads and frees a result: one output, or the named outputs.
func decodeResult(api *capi, result unsafe.Pointer, all bool) (*Tensor, map[string]*Tensor, error) {
	if result == nil {
		return nil, nil, &Error{Message: "the engine returned no result"}
	}
	defer api.resultFree(result)
	if message := api.resultError(result); message != nil {
		return nil, nil, &Error{Message: goString(message)}
	}
	count := api.resultCount(result)
	if !all {
		if count == 0 {
			return nil, nil, &Error{Message: "the pipeline emitted nothing"}
		}
		return readValue(api.resultValue(result, 0)), nil, nil
	}
	outputs := make(map[string]*Tensor, count)
	for i := uintptr(0); i < count; i++ {
		outputs[goString(api.resultName(result, i))] = readValue(api.resultValue(result, i))
	}
	return nil, outputs, nil
}
