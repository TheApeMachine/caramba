package tweaker

import "sync"

/*
Capsule wraps around a config value, or sequence of values, and
deals with providing the correct value. T can be any type.
Thread-safe for concurrent access.
*/
type Capsule[T any] struct {
	values []T
	idx    int
	mu     sync.Mutex
}

/*
NewCapsule creates a new Capsule instance with the provided values
and initializes the ring buffer index to 0.
*/
func NewCapsule[T any](values []T) *Capsule[T] {
	return &Capsule[T]{
		values: values,
		idx:    0,
	}
}

/*
Iter returns an iterator over the capsule's values in a round-robin fashion.
This implements the new Go 1.23 iterator pattern by reusing the existing
thread-safe ring buffer methods.

The iterator will continue until either:
1. The yield function returns false (early termination)
2. The capsule is empty (returns immediately)
*/
func (c *Capsule[T]) Iter() func(yield func(T) bool) {
	return func(yield func(T) bool) {
		c.mu.Lock()
		if len(c.values) == 0 {
			var zero T
			yield(zero)
			c.mu.Unlock()
			return
		}
		c.mu.Unlock()

		for {
			if !yield(c.Next()) {
				return
			}
		}
	}
}

/*
Next returns the next value in the ring buffer and advances the index.
If the buffer is empty, returns the zero value of type T.
Thread-safe for concurrent access.
*/
func (c *Capsule[T]) Next() T {
	c.mu.Lock()
	defer c.mu.Unlock()

	if len(c.values) == 0 {
		var zero T
		return zero
	}

	value := c.values[c.idx]
	c.idx = (c.idx + 1) % len(c.values)
	return value
}

/*
Peek returns the current value in the ring buffer without advancing the index.
If the buffer is empty, returns the zero value of type T.
Thread-safe for concurrent access.
*/
func (c *Capsule[T]) Peek() T {
	c.mu.Lock()
	defer c.mu.Unlock()

	if len(c.values) == 0 {
		var zero T
		return zero
	}

	return c.values[c.idx]
}

/*
Reset sets the ring buffer index back to the beginning of the sequence,
allowing the Next operation to start from the first value again.
Thread-safe for concurrent access.
*/
func (c *Capsule[T]) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.idx = 0
}
