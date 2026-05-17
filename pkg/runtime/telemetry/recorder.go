package telemetry

import (
	"sort"
	"sync"
	"time"
)

/*
Recorder is the runtime's instrumentation sink. The executor exposes
it through op.Context so telemetry ops (counter, histogram, trace,
scope) can push samples without coupling to a concrete backend.
Implementations decide where samples land — in-memory for tests,
JSONL for ledger writes, OTLP for production telemetry.
*/
type Recorder interface {
	EnterScope(name string)
	ExitScope()
	IncrementCounter(name string, delta float64)
	RecordHistogram(name string, value float64)
	RecordTensor(name string, values []float64, shape []int)
	Event(name string, fields map[string]any)
}

/*
Sample is one recorded telemetry datum.
*/
type Sample struct {
	Kind   string
	Name   string
	Scope  []string
	Value  float64
	Values []float64
	Shape  []int
	Fields map[string]any
	Time   time.Time
}

/*
InMemory is the default Recorder used by tests and short-lived
research runs. It keeps every sample in a sorted slice for easy
inspection. Snapshot returns a copy so callers can iterate without
holding the lock.
*/
type InMemory struct {
	mu       sync.Mutex
	scope    []string
	counters map[string]float64
	samples  []Sample
}

func NewInMemory() *InMemory {
	return &InMemory{counters: map[string]float64{}}
}

func (recorder *InMemory) EnterScope(name string) {
	recorder.mu.Lock()
	defer recorder.mu.Unlock()

	recorder.scope = append(recorder.scope, name)
}

func (recorder *InMemory) ExitScope() {
	recorder.mu.Lock()
	defer recorder.mu.Unlock()

	if len(recorder.scope) == 0 {
		return
	}

	recorder.scope = recorder.scope[:len(recorder.scope)-1]
}

func (recorder *InMemory) IncrementCounter(name string, delta float64) {
	recorder.mu.Lock()
	defer recorder.mu.Unlock()

	recorder.counters[name] += delta
	recorder.samples = append(recorder.samples, Sample{
		Kind:  "counter",
		Name:  name,
		Scope: append([]string(nil), recorder.scope...),
		Value: recorder.counters[name],
		Time:  time.Now(),
	})
}

func (recorder *InMemory) RecordHistogram(name string, value float64) {
	recorder.mu.Lock()
	defer recorder.mu.Unlock()

	recorder.samples = append(recorder.samples, Sample{
		Kind:  "histogram",
		Name:  name,
		Scope: append([]string(nil), recorder.scope...),
		Value: value,
		Time:  time.Now(),
	})
}

func (recorder *InMemory) RecordTensor(name string, values []float64, shape []int) {
	recorder.mu.Lock()
	defer recorder.mu.Unlock()

	recorder.samples = append(recorder.samples, Sample{
		Kind:   "tensor",
		Name:   name,
		Scope:  append([]string(nil), recorder.scope...),
		Values: append([]float64(nil), values...),
		Shape:  append([]int(nil), shape...),
		Time:   time.Now(),
	})
}

// Event records an arbitrary key/value event. The fields map is
// shallow-copied: top-level keys cannot be mutated by the caller after
// the call returns, but any nested mutable values (maps, slices, or
// pointer-bearing structs) remain shared with the caller. Callers
// passing mutable nested values must treat them as immutable after
// the call, or pass their own deep copy.
func (recorder *InMemory) Event(name string, fields map[string]any) {
	recorder.mu.Lock()
	defer recorder.mu.Unlock()

	fieldsCopy := map[string]any{}

	for key, value := range fields {
		fieldsCopy[key] = value
	}

	recorder.samples = append(recorder.samples, Sample{
		Kind:   "event",
		Name:   name,
		Scope:  append([]string(nil), recorder.scope...),
		Fields: fieldsCopy,
		Time:   time.Now(),
	})
}

/*
Snapshot returns a copy of the recorded samples ordered by insertion.
*/
func (recorder *InMemory) Snapshot() []Sample {
	recorder.mu.Lock()
	defer recorder.mu.Unlock()

	out := make([]Sample, len(recorder.samples))
	copy(out, recorder.samples)

	return out
}

/*
Counter reads the current accumulated counter value for diagnostics.
*/
func (recorder *InMemory) Counter(name string) float64 {
	recorder.mu.Lock()
	defer recorder.mu.Unlock()

	return recorder.counters[name]
}

/*
CounterNames returns every counter name observed so far in sorted
order.
*/
func (recorder *InMemory) CounterNames() []string {
	recorder.mu.Lock()
	defer recorder.mu.Unlock()

	names := make([]string, 0, len(recorder.counters))

	for name := range recorder.counters {
		names = append(names, name)
	}

	sort.Strings(names)

	return names
}

/*
NoOp is the recorder the executor uses when no telemetry sink is
configured. It satisfies the interface without allocating.
*/
type NoOp struct{}

func (NoOp) EnterScope(name string)                                {}
func (NoOp) ExitScope()                                            {}
func (NoOp) IncrementCounter(name string, delta float64)           {}
func (NoOp) RecordHistogram(name string, value float64)            {}
func (NoOp) RecordTensor(name string, values []float64, shape []int) {}
func (NoOp) Event(name string, fields map[string]any)              {}
