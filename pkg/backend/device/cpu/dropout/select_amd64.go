//go:build amd64

package dropout

func DropoutFloat32Native(
	dst, src []float32,
	seedState *[4]uint32,
	keepProb float32,
) {
	if len(src) == 0 {
		return
	}

	dropoutF32Kernel(&dst[0], &src[0], len(src), seedState, keepProb)
}

var dropoutF32Funcs = []f32DropoutKernelImpl{
	{DropoutF32Generic, "generic", true},
}
