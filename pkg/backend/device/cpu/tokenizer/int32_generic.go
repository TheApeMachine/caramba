package tokenizer

import "unsafe"

func packInt32Generic(dst, src *int32, count int) {
	PackInt32Scalar(unsafe.Slice(dst, count), unsafe.Slice(src, count))
}
