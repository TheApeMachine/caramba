package state

import (
	"context"
	"encoding/binary"
	"fmt"
	"sync"
)

/*
Counter is the canonical integer state used for runtime indices:
decoder position, training step, iteration counters. Increment and
Set are the two mutations; the runtime program decides which to use.
*/
type Counter struct {
	mu      sync.Mutex
	id      string
	initial int
	value   int
}

func newCounter(id string, initial int) *Counter {
	return &Counter{id: id, initial: initial, value: initial}
}

func newCounterFromConfig(id string, config map[string]any) (State, error) {
	initial, err := intFromConfig(config, "initial")

	if err != nil {
		return nil, err
	}

	return newCounter(id, initial), nil
}

func (counter *Counter) ID() string {
	return counter.id
}

func (counter *Counter) Type() string {
	return "counter"
}

func (counter *Counter) Reset(ctx context.Context) error {
	counter.mu.Lock()
	defer counter.mu.Unlock()

	counter.value = counter.initial

	return nil
}

/*
Value returns the current counter value.
*/
func (counter *Counter) Value() int {
	counter.mu.Lock()
	defer counter.mu.Unlock()

	return counter.value
}

/*
Set replaces the counter value.
*/
func (counter *Counter) Set(value int) {
	counter.mu.Lock()
	defer counter.mu.Unlock()

	counter.value = value
}

/*
Increment adds delta to the counter and returns the new value.
*/
func (counter *Counter) Increment(delta int) int {
	counter.mu.Lock()
	defer counter.mu.Unlock()

	counter.value += delta

	return counter.value
}

func (counter *Counter) Snapshot(ctx context.Context) (Snapshot, error) {
	payload := make([]byte, 8)
	binary.LittleEndian.PutUint64(payload, uint64(int64(counter.Value())))

	return Snapshot{
		StateID: counter.id,
		Type:    counter.Type(),
		Schema:  "int64-le",
		Payload: payload,
	}, nil
}

func (counter *Counter) Restore(ctx context.Context, snapshot Snapshot) error {
	if snapshot.Schema != "int64-le" {
		return fmt.Errorf("counter: unsupported snapshot schema %q", snapshot.Schema)
	}

	if len(snapshot.Payload) != 8 {
		return fmt.Errorf("counter: payload length %d != 8", len(snapshot.Payload))
	}

	value := int(int64(binary.LittleEndian.Uint64(snapshot.Payload)))
	counter.Set(value)

	return nil
}

func (counter *Counter) Inspect(ctx context.Context) (Inspection, error) {
	return Inspection{
		StateID: counter.id,
		Type:    counter.Type(),
		Values: map[string]any{
			"value":   counter.Value(),
			"initial": counter.initial,
		},
	}, nil
}
