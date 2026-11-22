package main

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// wav is a 16-bit mono WAV file of samples at 44.1 kHz.
func wav(samples []int16) []byte {
	data := make([]byte, 44+2*len(samples))
	copy(data[0:], "RIFF")
	binary.LittleEndian.PutUint32(data[4:], uint32(36+2*len(samples)))
	copy(data[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(data[16:], 16)
	binary.LittleEndian.PutUint16(data[20:], 1) // PCM
	binary.LittleEndian.PutUint16(data[22:], 1) // mono
	binary.LittleEndian.PutUint32(data[24:], 44100)
	binary.LittleEndian.PutUint32(data[28:], 44100*2)
	binary.LittleEndian.PutUint16(data[32:], 2)
	binary.LittleEndian.PutUint16(data[34:], 16)
	copy(data[36:], "data")
	binary.LittleEndian.PutUint32(data[40:], uint32(2*len(samples)))
	for i, s := range samples {
		binary.LittleEndian.PutUint16(data[44+2*i:], uint16(s))
	}
	return data
}

func TestLouder(t *testing.T) {
	t.Chdir("..")
	samples := make([]int16, 4410)
	for i := range samples {
		samples[i] = int16(8000 * math.Sin(float64(i)/7))
	}
	dir := t.TempDir()
	in, out := filepath.Join(dir, "in.wav"), filepath.Join(dir, "out.wav")
	os.WriteFile(in, wav(samples), 0o644)
	if err := louder(in, out); err != nil {
		if os.Getenv("MORFLOW_REQUIRE_ENGINE") != "1" {
			t.Skip(err)
		}
		t.Fatal(err)
	}
	result, _ := os.ReadFile(out)
	if string(result[:4]) != "RIFF" || len(result) != 44+2*len(samples) {
		t.Fatalf("got %d bytes starting %q", len(result), result[:4])
	}
	for i, s := range samples {
		got := int16(binary.LittleEndian.Uint16(result[44+2*i:]))
		if math.Abs(float64(got)-2*float64(s)) > 2 {
			t.Fatalf("sample %d: got %d, want about %d", i, got, 2*int(s))
		}
	}
	if err := louder(filepath.Join(dir, "missing.wav"), out); err == nil {
		t.Fatal("a missing input should fail")
	}
	if err := louder(in, filepath.Join(dir, "no", "such", "folder.wav")); err == nil {
		t.Fatal("an unwritable output should fail")
	}
}
