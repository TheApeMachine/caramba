package sampling

import (
	"testing"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func TestGreedySampleFloat32Native(t *testing.T) {
	logits := []float32{0.1, 0.2, 0.9, 0.3, 0.4}

	if got := GreedySampleFloat32Native(logits); got != 2 {
		t.Fatalf("greedy index=%d want=2", got)
	}
}

func TestGreedySample(t *testing.T) {
	logits := []float32{0.1, 0.2, 0.9, 0.3, 0.4}

	token := GreedySample(unsafe.Pointer(&logits[0]), len(logits), dtype.Float32)

	if token != 2 {
		t.Fatalf("token=%d want=2", token)
	}
}
