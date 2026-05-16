package kv

import (
	"fmt"
	"sync"
)

/*
Cache stores per-attention-node key/value tensors for incremental decoder execution.
*/
type Cache struct {
	mu       sync.Mutex
	epoch    uint64
	capacity int
	entries  map[string]*Entry
}

/*
Entry is the accumulated key/value state for one attention node.
Shape is [batch, heads, cached_tokens, head_dim].
*/
type Entry struct {
	Shape []int
	Key   []float64
	Value []float64
}

/*
NewCache instantiates an empty decoder KV cache.
*/
func NewCache() *Cache {
	return &Cache{
		epoch:   1,
		entries: make(map[string]*Entry),
	}
}

/*
Reset clears all accumulated key/value state.
*/
func (cache *Cache) Reset() {
	if cache == nil {
		return
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()

	clear(cache.entries)
	cache.epoch++
}

/*
SetCapacity records the intended token capacity for backend-resident KV stores.
Host Append remains dynamically sized; accelerated backends use this to
preallocate once and write decode chunks in place.
*/
func (cache *Cache) SetCapacity(capacity int) error {
	if cache == nil {
		return fmt.Errorf("kv cache: cache is required")
	}

	if capacity < 0 {
		return fmt.Errorf("kv cache: capacity must be non-negative")
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()

	cache.capacity = capacity

	return nil
}

/*
Capacity returns the intended token capacity for backend-resident KV stores.
*/
func (cache *Cache) Capacity() int {
	if cache == nil {
		return 0
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()

	return cache.capacity
}

/*
Epoch returns the cache reset generation.
Backend-resident caches use it to invalidate local tensors without copying K/V
through host memory.
*/
func (cache *Cache) Epoch() uint64 {
	if cache == nil {
		return 0
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()

	return cache.epoch
}

/*
Append adds a key/value chunk for nodeID and returns the full cached tensors.
*/
func (cache *Cache) Append(
	nodeID string,
	shape []int,
	key []float64,
	value []float64,
) ([]float64, []float64, []int, error) {
	if cache == nil {
		return nil, nil, nil, fmt.Errorf("kv cache: cache is required")
	}

	if len(shape) != 4 {
		return nil, nil, nil, fmt.Errorf("kv cache: expected rank 4 shape, got %d", len(shape))
	}

	if shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 || shape[3] <= 0 {
		return nil, nil, nil, fmt.Errorf("kv cache: shape dimensions must be positive")
	}

	chunkLength := shapeLength(shape)

	if len(key) != chunkLength || len(value) != chunkLength {
		return nil, nil, nil, fmt.Errorf(
			"kv cache: key/value lengths must match shape length %d, got %d and %d",
			chunkLength,
			len(key),
			len(value),
		)
	}

	cache.mu.Lock()
	defer cache.mu.Unlock()

	entry := cache.entries[nodeID]

	if entry == nil {
		entry = &Entry{Shape: append([]int(nil), shape...)}
		entry.Shape[2] = 0
		cache.entries[nodeID] = entry
	}

	if err := entry.validateAppend(shape); err != nil {
		return nil, nil, nil, err
	}

	entry.appendChunk(shape, key, value)

	return entry.Key, entry.Value, append([]int(nil), entry.Shape...), nil
}

func (entry *Entry) validateAppend(shape []int) error {
	if len(entry.Shape) == 0 {
		entry.Shape = append([]int(nil), shape...)
		entry.Shape[2] = 0

		return nil
	}

	if len(entry.Shape) != len(shape) {
		return fmt.Errorf("kv cache: cached shape rank changed from %d to %d", len(entry.Shape), len(shape))
	}

	for _, dimension := range []int{0, 1, 3} {
		if entry.Shape[dimension] == shape[dimension] {
			continue
		}

		return fmt.Errorf(
			"kv cache: shape dimension %d changed from %d to %d",
			dimension,
			entry.Shape[dimension],
			shape[dimension],
		)
	}

	return nil
}

func (entry *Entry) appendChunk(shape []int, key []float64, value []float64) {
	entry.Key = appendTokenDimension(entry.Key, key, entry.Shape, shape)
	entry.Value = appendTokenDimension(entry.Value, value, entry.Shape, shape)
	entry.Shape[2] += shape[2]
}

func appendTokenDimension(
	existing []float64,
	chunk []float64,
	existingShape []int,
	chunkShape []int,
) []float64 {
	batches := existingShape[0]
	heads := existingShape[1]
	existingTokens := existingShape[2]
	chunkTokens := chunkShape[2]
	headDim := existingShape[3]
	totalTokens := existingTokens + chunkTokens
	output := make([]float64, batches*heads*totalTokens*headDim)

	for batchIndex := range batches {
		for headIndex := range heads {
			existingOffset := ((batchIndex*heads + headIndex) * existingTokens) * headDim
			chunkOffset := ((batchIndex*heads + headIndex) * chunkTokens) * headDim
			outputOffset := ((batchIndex*heads + headIndex) * totalTokens) * headDim
			existingEnd := existingOffset + existingTokens*headDim
			chunkEnd := chunkOffset + chunkTokens*headDim
			outputChunkOffset := outputOffset + existingTokens*headDim

			copy(output[outputOffset:outputChunkOffset], existing[existingOffset:existingEnd])
			copy(output[outputChunkOffset:outputChunkOffset+chunkTokens*headDim], chunk[chunkOffset:chunkEnd])
		}
	}

	return output
}

func shapeLength(shape []int) int {
	length := 1

	for _, dimension := range shape {
		length *= dimension
	}

	return length
}
