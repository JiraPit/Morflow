package morflow

import (
	"fmt"
	"unsafe"
)

// TensorInput is a tensor, image or audio input with an explicit shape.
type TensorInput struct {
	// Data holds the elements: a []float32, []float64, []int8, []int16,
	// []int32, []int64, []uint32, []uint64, or a []byte with the raw element
	// bytes of any dtype.
	Data any
	// Shape lists the dimensions, outermost first.
	Shape []int
	// DType is the element type: "f32", "f64", "u8", "i8", "i16", "i32",
	// "i64", "u32" or "u64". When empty it follows Data's type, and a []byte
	// is "f32" unless you say otherwise.
	DType string
	// PayloadType is "tensor" (the default), "image" or "audio". Nothing is
	// inferred: a rank-3 tensor is a plain tensor unless you ask for "image".
	PayloadType string
	// ColorSpace is the color space of an image: "grayscale", "rgb", "rgba",
	// "bgr" or "bgra". When empty it follows the channel count.
	ColorSpace string
	// SampleRate is the sample rate of audio in Hz. Zero means 44100.
	SampleRate int
	// Channels is the channel count of audio. Zero means the leading
	// dimension.
	Channels int
	// Layout is "hwc" or "chw" for images and "planar" or "interleaved" for
	// audio. When empty it is "hwc" or "planar".
	Layout string
}

// Image is an image input of height rows of width pixels, each with
// channels values.
func Image(data any, height, width, channels int) TensorInput {
	return TensorInput{Data: data, Shape: []int{height, width, channels}, PayloadType: "image"}
}

// Audio is planar audio: one row of samples per channel, so samples holds
// channels rows one after another.
func Audio(samples []float32, channels, sampleRate int) TensorInput {
	frames := 0
	if channels > 0 {
		frames = len(samples) / channels
	}
	return TensorInput{Data: samples, Shape: []int{channels, frames}, PayloadType: "audio", SampleRate: sampleRate, Channels: channels}
}

// Tensor is a value a pipeline emitted. A Composite has DType "composite"
// and its parts in Items.
type Tensor struct {
	// Shape lists the dimensions, outermost first.
	Shape []int
	// DType is the element type, such as "f32", or "composite".
	DType string
	// Data holds the raw element bytes, in native byte order.
	Data []byte
	// Items holds the parts of a Composite, and is nil otherwise.
	Items []*Tensor
}

// IsComposite reports whether the tensor is a Composite of several values.
func (t *Tensor) IsComposite() bool { return t.DType == "composite" }

// Len is the number of elements.
func (t *Tensor) Len() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// Bytes returns the raw element bytes. For bytes a pipeline emits, such as a
// WAV file from to_wav, this is the file.
func (t *Tensor) Bytes() []byte { return t.Data }

// The typed accessors read the raw bytes as elements of their type, copied
// into a new slice. Check DType first: they do not convert between types.

func (t *Tensor) Float32s() []float32 { return elements[float32](t.Data) }
func (t *Tensor) Float64s() []float64 { return elements[float64](t.Data) }
func (t *Tensor) Int8s() []int8       { return elements[int8](t.Data) }
func (t *Tensor) Int16s() []int16     { return elements[int16](t.Data) }
func (t *Tensor) Int32s() []int32     { return elements[int32](t.Data) }
func (t *Tensor) Int64s() []int64     { return elements[int64](t.Data) }
func (t *Tensor) Uint8s() []uint8     { return append([]uint8(nil), t.Data...) }
func (t *Tensor) Uint32s() []uint32   { return elements[uint32](t.Data) }
func (t *Tensor) Uint64s() []uint64   { return elements[uint64](t.Data) }

// Values returns the elements as the slice type that matches DType, such as
// a []float32 for "f32".
func (t *Tensor) Values() (any, error) {
	switch t.DType {
	case "f32":
		return t.Float32s(), nil
	case "f64":
		return t.Float64s(), nil
	case "u8":
		return t.Uint8s(), nil
	case "i8":
		return t.Int8s(), nil
	case "i16":
		return t.Int16s(), nil
	case "i32":
		return t.Int32s(), nil
	case "i64":
		return t.Int64s(), nil
	case "u32":
		return t.Uint32s(), nil
	case "u64":
		return t.Uint64s(), nil
	}
	return nil, fmt.Errorf("morflow: a %q tensor has no element slice", t.DType)
}

func (t *Tensor) String() string {
	if t.IsComposite() {
		return fmt.Sprintf("Tensor(composite of %d)", len(t.Items))
	}
	return fmt.Sprintf("Tensor(%s %v)", t.DType, t.Shape)
}

type element interface {
	~float32 | ~float64 | ~int8 | ~int16 | ~int32 | ~int64 | ~uint32 | ~uint64
}

func elements[T element](data []byte) []T {
	var zero T
	size := int(unsafe.Sizeof(zero))
	out := make([]T, len(data)/size)
	if len(out) > 0 {
		copy(unsafe.Slice((*byte)(unsafe.Pointer(&out[0])), len(out)*size), data)
	}
	return out
}

// bytesOf views a slice's elements as bytes, without copying.
func bytesOf[T element](s []T) []byte {
	if len(s) == 0 {
		return nil
	}
	var zero T
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*int(unsafe.Sizeof(zero)))
}
