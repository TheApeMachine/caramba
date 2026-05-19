package masking

import "unsafe"

func ApplyMaskFloat32Native(input, mask, output unsafe.Pointer, count int) {
	if count == 0 {
		return
	}

	applyMaskF32Kernel(
		(*float32)(input),
		(*float32)(mask),
		(*float32)(output),
		count,
	)
}

func CausalMaskFloat32Native(output unsafe.Pointer, seqQ, seqK int) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	causalMaskF32Kernel((*float32)(output), seqQ, seqK)
}

func ALiBiBiasFloat32Native(scores, slope, output unsafe.Pointer, seqQ, seqK int) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	alibiBiasF32Kernel(
		(*float32)(scores),
		(*float32)(slope),
		(*float32)(output),
		seqQ,
		seqK,
	)
}
