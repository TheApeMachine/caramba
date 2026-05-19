package pool

import "sync"

var float32BufferPool sync.Pool

func BorrowFloat32Buffer(length int) []float32 {
	pooled, ok := float32BufferPool.Get().(*[]float32)

	if !ok || pooled == nil || cap(*pooled) < length {
		buffer := make([]float32, length)
		return buffer
	}

	buffer := (*pooled)[:length]
	return buffer
}

func ReleaseFloat32Buffer(buffer []float32) {
	if buffer == nil {
		return
	}

	float32BufferPool.Put(&buffer)
}
