//go:build !amd64 && !arm64 && !386

package activation

var (
	swiGLUTensorsFuncs      = []gatedTensorsKernelImpl{{SwiGLUTensorsF32Generic, "generic", true}}
	linGLUTensorsFuncs      = []gatedTensorsKernelImpl{{LinGLUTensorsF32Generic, "generic", true}}
	reGLUTensorsFuncs       = []gatedTensorsKernelImpl{{ReGLUTensorsF32Generic, "generic", true}}
	gluTensorsFuncs         = []gatedTensorsKernelImpl{{GLUTensorsF32Generic, "generic", true}}
	siGLUTensorsFuncs       = []gatedTensorsKernelImpl{{SiGLUTensorsF32Generic, "generic", true}}
	seGLUTensorsFuncs       = []gatedTensorsKernelImpl{{SeGLUTensorsF32Generic, "generic", true}}
	geGLUTensorsFuncs       = []gatedTensorsKernelImpl{{GeGLUTensorsF32Generic, "generic", true}}
	geGLUTanhTensorsFuncs   = []gatedTensorsKernelImpl{{GeGLUTanhTensorsF32Generic, "generic", true}}
)
