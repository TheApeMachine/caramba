package state

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"sync"
)

/*
Tensor is the runtime's host-side dense float64 tensor state. It is
the default backing for resident activations, latents, embeddings,
and intermediate runtime values. Backend-resident variants attach
through the Bind path defined in the BackendState contract that will
land alongside the executor's residency planner.
*/
type Tensor struct {
	mu     sync.Mutex
	id     string
	shape  []int
	values []float64
}

func newTensor(id string, shape []int) *Tensor {
	allocated := tensorCapacity(shape)

	return &Tensor{
		id:     id,
		shape:  append([]int(nil), shape...),
		values: make([]float64, allocated),
	}
}

func newTensorFromConfig(id string, config map[string]any) (State, error) {
	shape, err := intSliceFromConfig(config, "shape")

	if err != nil {
		return nil, err
	}

	if len(shape) == 0 {
		return nil, fmt.Errorf("tensor: %q requires a non-empty shape", id)
	}

	tensor := newTensor(id, shape)
	initial, err := stringFromConfig(config, "init")

	if err != nil {
		return nil, err
	}

	seed, err := int64FromConfig(config, "seed")

	if err != nil {
		return nil, err
	}

	if err := tensor.applyInitializer(initial, uint64(seed)); err != nil {
		return nil, err
	}

	return tensor, nil
}

func (tensor *Tensor) ID() string {
	return tensor.id
}

func (tensor *Tensor) Type() string {
	return "tensor"
}

func (tensor *Tensor) Reset(ctx context.Context) error {
	tensor.mu.Lock()
	defer tensor.mu.Unlock()

	for index := range tensor.values {
		tensor.values[index] = 0
	}

	return nil
}

/*
Shape returns a copy of the tensor's current shape.
*/
func (tensor *Tensor) Shape() []int {
	tensor.mu.Lock()
	defer tensor.mu.Unlock()

	return append([]int(nil), tensor.shape...)
}

/*
Values returns a copy of the tensor's current contents.
*/
func (tensor *Tensor) Values() []float64 {
	tensor.mu.Lock()
	defer tensor.mu.Unlock()

	return append([]float64(nil), tensor.values...)
}

/*
Set replaces the tensor's shape and contents. Length must match the
new shape's product.
*/
func (tensor *Tensor) Set(shape []int, values []float64) error {
	if len(shape) == 0 {
		return fmt.Errorf("tensor: shape must be non-empty")
	}

	expected := tensorCapacity(shape)

	if expected != len(values) {
		return fmt.Errorf(
			"tensor: shape product %d != value length %d",
			expected,
			len(values),
		)
	}

	tensor.mu.Lock()
	tensor.shape = append([]int(nil), shape...)
	tensor.values = append([]float64(nil), values...)
	tensor.mu.Unlock()

	return nil
}

func (tensor *Tensor) Snapshot(ctx context.Context) (Snapshot, error) {
	tensor.mu.Lock()
	defer tensor.mu.Unlock()

	header := make([]byte, 4+8*len(tensor.shape))
	binary.LittleEndian.PutUint32(header[0:4], uint32(len(tensor.shape)))

	for index, dim := range tensor.shape {
		binary.LittleEndian.PutUint64(header[4+index*8:], uint64(int64(dim)))
	}

	body := make([]byte, 8*len(tensor.values))

	for index, value := range tensor.values {
		binary.LittleEndian.PutUint64(body[index*8:], math.Float64bits(value))
	}

	return Snapshot{
		StateID: tensor.id,
		Type:    tensor.Type(),
		Schema:  "float64-le-shape-header",
		Payload: append(header, body...),
	}, nil
}

func (tensor *Tensor) Restore(ctx context.Context, snapshot Snapshot) error {
	if snapshot.Schema != "float64-le-shape-header" {
		return fmt.Errorf("tensor: unsupported snapshot schema %q", snapshot.Schema)
	}

	if len(snapshot.Payload) < 4 {
		return fmt.Errorf("tensor: payload truncated header")
	}

	rank := int(binary.LittleEndian.Uint32(snapshot.Payload[0:4]))
	headerSize := 4 + 8*rank

	if len(snapshot.Payload) < headerSize {
		return fmt.Errorf("tensor: payload missing %d shape entries", rank)
	}

	shape := make([]int, rank)

	for index := range shape {
		shape[index] = int(int64(binary.LittleEndian.Uint64(snapshot.Payload[4+index*8:])))
	}

	expected := tensorCapacity(shape)
	bodyBytes := len(snapshot.Payload) - headerSize

	if bodyBytes != 8*expected {
		return fmt.Errorf(
			"tensor: payload body bytes %d != %d (shape product %d * 8)",
			bodyBytes,
			8*expected,
			expected,
		)
	}

	values := make([]float64, expected)

	for index := range values {
		offset := headerSize + index*8
		values[index] = math.Float64frombits(binary.LittleEndian.Uint64(snapshot.Payload[offset:]))
	}

	return tensor.Set(shape, values)
}

func (tensor *Tensor) Inspect(ctx context.Context) (Inspection, error) {
	tensor.mu.Lock()
	defer tensor.mu.Unlock()

	shapeCopy := append([]int(nil), tensor.shape...)

	return Inspection{
		StateID: tensor.id,
		Type:    tensor.Type(),
		Values: map[string]any{
			"shape":  shapeCopy,
			"length": len(tensor.values),
		},
	}, nil
}

func (tensor *Tensor) applyInitializer(name string, seed uint64) error {
	switch name {
	case "", "zeros":
		return nil
	case "gaussian":
		fillGaussian(tensor.values, seed)

		return nil
	}

	return fmt.Errorf("tensor: unknown initializer %q (built-ins: zeros, gaussian)", name)
}

func fillGaussian(values []float64, seed uint64) {
	if seed == 0 {
		seed = 0x9e3779b97f4a7c15
	}

	source := rand.NewChaCha8(gaussianSeed(seed))
	stream := rand.New(source)

	for index := range values {
		values[index] = stream.NormFloat64()
	}
}

func gaussianSeed(seed uint64) [32]byte {
	var out [32]byte
	binary.LittleEndian.PutUint64(out[0:8], seed)
	binary.LittleEndian.PutUint64(out[8:16], seed^0x9e3779b97f4a7c15)
	binary.LittleEndian.PutUint64(out[16:24], seed^0xbf58476d1ce4e5b9)
	binary.LittleEndian.PutUint64(out[24:32], seed^0x94d049bb133111eb)

	return out
}

func tensorCapacity(shape []int) int {
	product := 1

	for _, dim := range shape {
		if dim <= 0 {
			return 0
		}

		product *= dim
	}

	return product
}
