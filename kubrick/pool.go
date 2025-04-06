package kubrick

import (
	"sync"
)

const (
	smallBufferSize  = 80 * 5    // 5 lines at 80 chars each
	mediumBufferSize = 80 * 24   // 24 lines at 80 chars each
	largeBufferSize  = 200 * 100 // 100 lines at 200 chars each
)

// BufferPool manages memory pools for different buffer sizes
type BufferPool struct {
	small  sync.Pool
	medium sync.Pool
	large  sync.Pool
}

// NewBufferPool creates a new buffer pool
func NewBufferPool() *BufferPool {
	return &BufferPool{
		small: sync.Pool{
			New: func() interface{} {
				buf := make([][]rune, 5)
				return &buf
			},
		},
		medium: sync.Pool{
			New: func() interface{} {
				buf := make([][]rune, 24)
				return &buf
			},
		},
		large: sync.Pool{
			New: func() interface{} {
				buf := make([][]rune, 100)
				return &buf
			},
		},
	}
}

// GetBuffer gets a buffer of appropriate size
func (p *BufferPool) GetBuffer(rows, cols int) [][]rune {
	size := rows * cols
	var buf *[][]rune

	switch {
	case size <= smallBufferSize:
		buf = p.small.Get().(*[][]rune)
	case size <= mediumBufferSize:
		buf = p.medium.Get().(*[][]rune)
	default:
		buf = p.large.Get().(*[][]rune)
	}

	// Ensure buffer has correct dimensions
	if len(*buf) < rows {
		*buf = make([][]rune, rows)
	}
	for i := range *buf {
		if len((*buf)[i]) < cols {
			(*buf)[i] = make([]rune, cols)
		}
	}

	return *buf
}

// PutBuffer returns a buffer to the pool
func (p *BufferPool) PutBuffer(buf [][]rune) {
	size := len(buf) * len(buf[0])

	// Clear buffer before returning to pool
	for i := range buf {
		for j := range buf[i] {
			buf[i][j] = ' '
		}
	}

	bufPtr := &buf
	switch {
	case size <= smallBufferSize:
		p.small.Put(bufPtr)
	case size <= mediumBufferSize:
		p.medium.Put(bufPtr)
	default:
		p.large.Put(bufPtr)
	}
}

// Global buffer pool instance
var globalBufferPool = NewBufferPool()
