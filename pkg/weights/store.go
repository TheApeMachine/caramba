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
[batch, in_features] × [in_features, out_features] with the resident
transpose materialized once.
The transposed handle is materialized once per name and cached.
*/
type Store struct {
	memory  tensor.Backend
	entries map[string]tensorEntry

	mu                   sync.Mutex
	cache                map[string]tensor.Tensor
	sliceCache           map[weightSliceKey]tensor.Tensor
	transposedCache      map[string]tensor.Tensor
	transposedSliceCache map[weightSliceKey]tensor.Tensor
}

type weightSliceKey struct {
	name  string
	axis  string
	start int64
	end   int64
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
		memory:               memory,
		entries:              make(map[string]tensorEntry),
		cache:                make(map[string]tensor.Tensor),
		sliceCache:           make(map[weightSliceKey]tensor.Tensor),
		transposedCache:      make(map[string]tensor.Tensor),
		transposedSliceCache: make(map[weightSliceKey]tensor.Tensor),
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
	buffer, sourceDType, shape, err := store.readEntryBuffer(name, entry)

	if err != nil {
		return nil, err
	}

	if sourceDType == dtype.Float32 {
		return store.memory.Upload(shape, dtype.Float32, buffer)
	}

	values, err := convert.BytesToFloat32(sourceDType, buffer)

	if err != nil {
		return nil, fmt.Errorf("weights store: convert %q: %w", name, err)
	}

	return store.memory.Upload(shape, dtype.Float32, float32ValuesToBytes(values))
}

func (store *Store) readEntryBuffer(
	name string,
	entry tensorEntry,
) ([]byte, dtype.DType, tensor.Shape, error) {
	file, err := os.Open(entry.path)

	if err != nil {
		return nil, dtype.Invalid, tensor.Shape{}, fmt.Errorf(
			"weights store: open %q: %w",
			entry.path,
			err,
		)
	}

	defer file.Close()

	start := entry.dataBase + entry.meta.DataOffsets[0]
	length := entry.meta.DataOffsets[1] - entry.meta.DataOffsets[0]

	if length < 0 {
		return nil, dtype.Invalid, tensor.Shape{}, fmt.Errorf(
			"weights store: tensor %q has negative length",
			name,
		)
	}

	buffer := make([]byte, length)

	if _, err := file.ReadAt(buffer, start); err != nil {
		return nil, dtype.Invalid, tensor.Shape{}, fmt.Errorf(
			"weights store: read %q: %w",
			name,
			err,
		)
	}

	sourceDType, err := dtype.Parse(entry.meta.DType)

	if err != nil {
		return nil, dtype.Invalid, tensor.Shape{}, fmt.Errorf(
			"weights store: parse dtype for %q: %w",
			name,
			err,
		)
	}

	dims := make([]int, len(entry.meta.Shape))

	for index, dimension := range entry.meta.Shape {
		dims[index] = int(dimension)
	}

	shape, err := tensor.NewShape(dims)

	if err != nil {
		return nil, dtype.Invalid, tensor.Shape{}, fmt.Errorf(
			"weights store: shape for %q: %w",
			name,
			err,
		)
	}

	return buffer, sourceDType, shape, nil
}

func (store *Store) loadFloat32Values(
	name string,
	entry tensorEntry,
) (tensor.Shape, []float32, error) {
	buffer, sourceDType, shape, err := store.readEntryBuffer(name, entry)

	if err != nil {
		return tensor.Shape{}, nil, err
	}

	if sourceDType != dtype.Float32 {
		values, err := convert.BytesToFloat32(sourceDType, buffer)

		if err != nil {
			return tensor.Shape{}, nil, fmt.Errorf("weights store: convert %q: %w", name, err)
		}

		return shape, values, nil
	}

	if len(buffer)%4 != 0 {
		return tensor.Shape{}, nil, fmt.Errorf(
			"weights store: tensor %q byte length %d is not divisible by 4",
			name,
			len(buffer),
		)
	}

	values := make([]float32, len(buffer)/4)

	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(buffer[index*4:]))
	}

	return shape, values, nil
}

func float32ValuesToBytes(values []float32) []byte {
	output := make([]byte, len(values)*4)

	for index, value := range values {
		binary.LittleEndian.PutUint32(output[index*4:], math.Float32bits(value))
	}

	return output
}

/*
LookupTransposed returns the row-major transpose of the named 2-D weight.
The result is materialized once and cached; subsequent calls reuse the
resident tensor. Non-2-D weights are rejected with a clear error so a
caller that asks for an inappropriate transpose surfaces the bug at
dispatch time instead of corrupting silently.
*/
func (store *Store) LookupTransposed(name string) (tensor.Tensor, error) {
	store.mu.Lock()

	if cached, ok := store.transposedCache[name]; ok {
		store.mu.Unlock()
		return cached, nil
	}

	store.mu.Unlock()

	entry, ok := store.entries[name]

	if !ok {
		return nil, execution.ErrWeightNotFound
	}

	shape, source, err := store.loadFloat32Values(name, entry)

	if err != nil {
		return nil, err
	}

	transposedShape, transposedBytes, err := transposeFloat32Tensor(name, shape, source)

	if err != nil {
		return nil, err
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
var _ execution.SliceLookup = (*Store)(nil)
var _ execution.TransposedSliceLookup = (*Store)(nil)
