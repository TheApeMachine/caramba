package shape

func CopyContiguousFloat32Native(dst, src []float32) {
	if len(dst) == 0 {
		return
	}

	copyContiguousF32Kernel(&dst[0], &src[0], len(dst))
}

func WhereFloat32Native(dst, positive, negative []float32, mask []byte) {
	if len(dst) == 0 {
		return
	}

	whereF32Kernel(&dst[0], &positive[0], &negative[0], mask, len(dst))
}

func MaskedFillFloat32Native(dst, input []float32, fill float32, mask []byte) {
	if len(dst) == 0 {
		return
	}

	maskedFillF32Kernel(&dst[0], &input[0], fill, mask, len(dst))
}
