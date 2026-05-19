package dropout

import "unsafe"

func runDropoutF32(dst, src unsafe.Pointer, count int, config DropoutConfig) {
	if config.Rate <= 0 {
		destination := unsafe.Slice((*float32)(dst), count)
		source := unsafe.Slice((*float32)(src), count)
		copy(destination, source)

		return
	}

	keepProb := float32(1.0 - config.Rate)
	seedState := DropoutSeedState(config.Seed)

	dropoutF32Kernel(
		(*float32)(dst),
		(*float32)(src),
		count,
		&seedState,
		keepProb,
	)
}
