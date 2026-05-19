//go:build !amd64 && !arm64

package activation

var (
	softmaxF32Funcs    = []f32KernelImpl{{SoftmaxF32Generic, "generic", true}}
	logSoftmaxF32Funcs = []f32KernelImpl{{LogSoftmaxF32Generic, "generic", true}}
)
