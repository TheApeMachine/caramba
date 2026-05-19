package convolution

import "sync"

var float32ScratchPool = sync.Pool{
	New: func() any {
		buffer := make([]float32, 0, 16384)
		return &buffer
	},
}

func BorrowFloat32Buffer(length int) []float32 {
	bufferPointer := float32ScratchPool.Get().(*[]float32)
	buffer := *bufferPointer

	if cap(buffer) < length {
		buffer = make([]float32, length)
	} else {
		buffer = buffer[:length]
	}

	return buffer
}

func ReleaseFloat32Buffer(buffer []float32) {
	buffer = buffer[:0]
	float32ScratchPool.Put(&buffer)
}
