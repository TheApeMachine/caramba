package state

import (
	"context"
	"encoding/binary"
	"fmt"
	"math/rand/v2"
	"sync"
)

/*
RNG is the runtime random-number-generator state. It is seedable,
snapshottable, and restorable, so a researcher can reproduce a run
exactly by recovering its RNG snapshots from the provenance ledger.
*/
type RNG struct {
	mu     sync.Mutex
	id     string
	seed   uint64
	stream uint64
	source *rand.ChaCha8
	random *rand.Rand
}

func newRNG(id string, seed uint64) *RNG {
	rng := &RNG{id: id, seed: seed}
	rng.source = rand.NewChaCha8(rng.seedArray())
	rng.random = rand.New(rng.source)

	return rng
}

func newRNGFromConfig(id string, config map[string]any) (State, error) {
	seed, err := int64FromConfig(config, "seed")

	if err != nil {
		return nil, err
	}

	return newRNG(id, uint64(seed)), nil
}

func (rng *RNG) ID() string {
	return rng.id
}

func (rng *RNG) Type() string {
	return "rng"
}

func (rng *RNG) Reset(ctx context.Context) error {
	rng.mu.Lock()
	defer rng.mu.Unlock()

	rng.stream = 0
	rng.source = rand.NewChaCha8(rng.seedArray())
	rng.random = rand.New(rng.source)

	return nil
}

/*
Float64 draws a uniform [0,1) sample.
*/
func (rng *RNG) Float64() float64 {
	rng.mu.Lock()
	defer rng.mu.Unlock()

	rng.stream++

	return rng.random.Float64()
}

/*
Intn returns a pseudo-random integer in [0, n).
*/
func (rng *RNG) Intn(upperBound int) int {
	if upperBound <= 0 {
		return 0
	}

	rng.mu.Lock()
	defer rng.mu.Unlock()

	rng.stream++

	return rng.random.IntN(upperBound)
}

/*
NormFloat64 returns a normally distributed float64 with mean 0 and
standard deviation 1.
*/
func (rng *RNG) NormFloat64() float64 {
	rng.mu.Lock()
	defer rng.mu.Unlock()

	rng.stream++

	return rng.random.NormFloat64()
}

func (rng *RNG) Snapshot(ctx context.Context) (Snapshot, error) {
	rng.mu.Lock()
	defer rng.mu.Unlock()

	payload := make([]byte, 16)
	binary.LittleEndian.PutUint64(payload[0:8], rng.seed)
	binary.LittleEndian.PutUint64(payload[8:16], rng.stream)

	return Snapshot{
		StateID: rng.id,
		Type:    rng.Type(),
		Schema:  "seed-stream-le",
		Payload: payload,
	}, nil
}

func (rng *RNG) Restore(ctx context.Context, snapshot Snapshot) error {
	if snapshot.Schema != "seed-stream-le" {
		return fmt.Errorf("rng: unsupported snapshot schema %q", snapshot.Schema)
	}

	if len(snapshot.Payload) != 16 {
		return fmt.Errorf("rng: payload length %d != 16", len(snapshot.Payload))
	}

	rng.mu.Lock()
	defer rng.mu.Unlock()

	rng.seed = binary.LittleEndian.Uint64(snapshot.Payload[0:8])
	stream := binary.LittleEndian.Uint64(snapshot.Payload[8:16])
	rng.source = rand.NewChaCha8(rng.seedArray())
	rng.random = rand.New(rng.source)
	rng.stream = 0

	// Advance the keystream under a single lock acquisition rather
	// than re-taking rng.mu inside each Float64 call. rand.ChaCha8
	// does not expose a seek primitive, so the loop is the only way
	// to reach the recorded stream offset.
	for index := uint64(0); index < stream; index++ {
		rng.random.Float64()
		rng.stream++
	}

	return nil
}

func (rng *RNG) Inspect(ctx context.Context) (Inspection, error) {
	rng.mu.Lock()
	defer rng.mu.Unlock()

	return Inspection{
		StateID: rng.id,
		Type:    rng.Type(),
		Values: map[string]any{
			"seed":   rng.seed,
			"stream": rng.stream,
		},
	}, nil
}

func (rng *RNG) seedArray() [32]byte {
	var seed [32]byte
	binary.LittleEndian.PutUint64(seed[0:8], rng.seed)
	binary.LittleEndian.PutUint64(seed[8:16], rng.seed^0x9e3779b97f4a7c15)
	binary.LittleEndian.PutUint64(seed[16:24], rng.seed^0xbf58476d1ce4e5b9)
	binary.LittleEndian.PutUint64(seed[24:32], rng.seed^0x94d049bb133111eb)

	return seed
}
