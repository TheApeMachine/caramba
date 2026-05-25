/*
Package weights wires safetensors archives into the puter execution backend
as a concrete WeightStore implementation.

When a manifest program includes `hf://<repo>` the orchestrator already
resolves config.json to generate the architecture YAML; the actual tensor
storage (model.safetensors, sharded *.safetensors, etc.) is downloaded by
this package and indexed lazily on first lookup. The dispatcher calls
WeightStore.Lookup(name) for each weighted node and receives a
host-resident Float32 tensor.
*/
package weights

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sync"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
	"github.com/theapemachine/manifesto/tensor"
	weightsindex "github.com/theapemachine/manifesto/weights"
	"github.com/theapemachine/puter/execution"
)

/*
tensorEntry records where one tensor lives in one safetensors archive.
*/
type tensorEntry struct {
	path     string
	dataBase int64
	meta     weightsindex.TensorMeta
}

/*
Store implements execution.WeightStore over one or more safetensors files.
Tensors are read on first lookup, converted to Float32 if necessary, and
cached. The store is safe for concurrent lookups.

The store also implements puter/execution.TransposedLookup so the
dispatcher can request the row-major transpose of a 2-D weight. HuggingFace
stores linear layers as [out_features, in_features] and the canonical PyTorch
forward computes y = x @ W.T; the dispatcher's generic Matmul kernel expects
row-major [inner, cols], so projection.linear consumers ask for the
transposed handle via LookupTransposed and the matmul operates as
[batch, in_features] × [in_features, out_features] without an inline copy.
The transposed handle is materialized once per name and cached.
*/
type Store struct {
	memory  tensor.Backend
	entries map[string]tensorEntry

	mu              sync.Mutex
	cache           map[string]tensor.Tensor
	transposedCache map[string]tensor.Tensor
}

/*
New indexes every supplied safetensors file and returns a Store ready to
serve Lookup calls. The host-side tensor.Backend is required so the store
can upload converted weight buffers into resident storage.
*/
func New(memory tensor.Backend, files []string) (*Store, error) {
	if memory == nil {
		return nil, fmt.Errorf("weights store: memory backend is required")
	}

	if len(files) == 0 {
		return nil, fmt.Errorf("weights store: at least one safetensors file is required")
	}

	store := &Store{
		memory:          memory,
		entries:         make(map[string]tensorEntry),
		cache:           make(map[string]tensor.Tensor),
		transposedCache: make(map[string]tensor.Tensor),
	}

	for _, path := range files {
		index, dataBase, err := weightsindex.IndexFile(path)

		if err != nil {
			return nil, fmt.Errorf("weights store: index %q: %w", path, err)
		}

		for name, meta := range index {
			store.entries[name] = tensorEntry{
				path:     path,
				dataBase: dataBase,
				meta:     meta,
			}
		}
	}

	return store, nil
}

/*
Names returns the set of tensor names known to the store. Useful for
diagnostics and for tests that want to assert that a checkpoint matches an
architecture's expected weight list.
*/
func (store *Store) Names() []string {
	names := make([]string, 0, len(store.entries))

	for name := range store.entries {
		names = append(names, name)
	}

	return names
}

/*
Lookup returns the host-resident tensor for the given checkpoint name. The
returned tensor is always Float32, regardless of the on-disk dtype, because
the device kernels currently consume Float32 exclusively (the FP-aware
upload path lands with the per-backend dtype rewrite).
*/
func (store *Store) Lookup(name string) (tensor.Tensor, error) {
	store.mu.Lock()

	if cached, ok := store.cache[name]; ok {
		store.mu.Unlock()
		return cached, nil
	}

	store.mu.Unlock()

	entry, ok := store.entries[name]

	if !ok {
		return nil, execution.ErrWeightNotFound
	}

	tensorOut, err := store.load(name, entry)

	if err != nil {
		return nil, err
	}

	store.mu.Lock()
	defer store.mu.Unlock()

	// Another goroutine may have loaded the same tensor while we were
	// reading; keep the first one we observe to avoid duplicating storage.
	if cached, ok := store.cache[name]; ok {
		return cached, nil
	}

	store.cache[name] = tensorOut

	return tensorOut, nil
}

func (store *Store) load(name string, entry tensorEntry) (tensor.Tensor, error) {
	file, err := os.Open(entry.path)

	if err != nil {
		return nil, fmt.Errorf("weights store: open %q: %w", entry.path, err)
	}

	defer file.Close()

	start := entry.dataBase + entry.meta.DataOffsets[0]
	length := entry.meta.DataOffsets[1] - entry.meta.DataOffsets[0]

	if length < 0 {
		return nil, fmt.Errorf("weights store: tensor %q has negative length", name)
	}

	buffer := make([]byte, length)

	if _, err := file.ReadAt(buffer, start); err != nil {
		return nil, fmt.Errorf("weights store: read %q: %w", name, err)
	}

	sourceDType, err := dtype.Parse(entry.meta.DType)

	if err != nil {
		return nil, fmt.Errorf("weights store: parse dtype for %q: %w", name, err)
	}

	dims := make([]int, len(entry.meta.Shape))

	for index, dimension := range entry.meta.Shape {
		dims[index] = int(dimension)
	}

	shape, err := tensor.NewShape(dims)

	if err != nil {
		return nil, fmt.Errorf("weights store: shape for %q: %w", name, err)
	}

	if sourceDType == dtype.Float32 {
		return store.memory.Upload(shape, dtype.Float32, buffer)
	}

	values, err := convert.BytesToFloat32(sourceDType, buffer)

	if err != nil {
		return nil, fmt.Errorf("weights store: convert %q: %w", name, err)
	}

	output := make([]byte, len(values)*4)

	for index, value := range values {
		binary.LittleEndian.PutUint32(output[index*4:], math.Float32bits(value))
	}

	return store.memory.Upload(shape, dtype.Float32, output)
}

/*
LookupTransposed returns the row-major transpose of the named 2-D weight.
The result is materialized once and cached; subsequent calls reuse the
resident tensor. Non-2-D weights are rejected with a clear error so a
caller that asks for an inappropriate transpose surfaces the bug at
dispatch time instead of corrupting silently.

Implementation note: today the transpose is performed host-side via
Float32Native — fine while caramba pins the device pool to tensor.Host
(see runProgram). When the dispatcher learns to route through resident
device buffers per puter/ARCHITECTURE.md §3.1, this method should
delegate to the device's shape.transpose / device.Backend pathway so the
copy stays on-device.
*/
func (store *Store) LookupTransposed(name string) (tensor.Tensor, error) {
	store.mu.Lock()

	if cached, ok := store.transposedCache[name]; ok {
		store.mu.Unlock()
		return cached, nil
	}

	store.mu.Unlock()

	original, err := store.Lookup(name)

	if err != nil {
		return nil, err
	}

	dims := original.Shape().Dims()

	if len(dims) != 2 {
		return nil, fmt.Errorf(
			"weights store: %q has rank %d, transpose requires rank 2",
			name, len(dims),
		)
	}

	rows, cols := dims[0], dims[1]

	source, err := original.Float32Native()

	if err != nil {
		return nil, fmt.Errorf("weights store: read %q for transpose: %w", name, err)
	}

	transposedBytes := make([]byte, rows*cols*4)

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		rowBase := rowIndex * cols

		for colIndex := 0; colIndex < cols; colIndex++ {
			outOffset := (colIndex*rows + rowIndex) * 4

			binary.LittleEndian.PutUint32(
				transposedBytes[outOffset:],
				math.Float32bits(source[rowBase+colIndex]),
			)
		}
	}

	transposedShape, err := tensor.NewShape([]int{cols, rows})

	if err != nil {
		return nil, fmt.Errorf("weights store: shape for transposed %q: %w", name, err)
	}

	transposed, err := store.memory.Upload(transposedShape, dtype.Float32, transposedBytes)

	if err != nil {
		return nil, fmt.Errorf("weights store: upload transposed %q: %w", name, err)
	}

	store.mu.Lock()
	defer store.mu.Unlock()

	if cached, ok := store.transposedCache[name]; ok {
		return cached, nil
	}

	store.transposedCache[name] = transposed

	return transposed, nil
}

var _ execution.WeightStore = (*Store)(nil)
var _ execution.TransposedLookup = (*Store)(nil)
