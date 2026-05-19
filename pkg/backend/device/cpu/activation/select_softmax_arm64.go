//go:build arm64

package activation

func SoftmaxF32NEON(dst, src *float32, count int)
func LogSoftmaxF32NEON(dst, src *float32, count int)

var (
	softmaxF32Funcs = []f32KernelImpl{
		{SoftmaxF32NEON, "neon", true},
		{SoftmaxF32Generic, "generic", true},
	}
	logSoftmaxF32Funcs = []f32KernelImpl{
		{LogSoftmaxF32NEON, "neon", true},
		{LogSoftmaxF32Generic, "generic", true},
	}
)
