package model

import (
	"math/rand"
	"sync"
)

var (
	randMu  sync.Mutex
	randSrc = rand.New(rand.NewSource(42)) //nolint:gosec
)

func newRand() *rand.Rand {
	randMu.Lock()
	seed := randSrc.Int63()
	randMu.Unlock()

	return rand.New(rand.NewSource(seed)) //nolint:gosec
}

func gaussianSlice(n int, scale float64) []float64 {
	rng := newRand()
	out := make([]float64, n)

	for idx := range out {
		out[idx] = rng.NormFloat64() * scale
	}

	return out
}
