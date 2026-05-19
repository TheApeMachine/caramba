//go:build !amd64 && !arm64 && !386

package activation

var (
	swiGLUPackedFuncs    = []gatedPackedKernelImpl{{SwiGLUPackedF32Generic, "generic", true}}
	linGLUPackedFuncs    = []gatedPackedKernelImpl{{LinGLUPackedF32Generic, "generic", true}}
	reGLUPackedFuncs     = []gatedPackedKernelImpl{{ReGLUPackedF32Generic, "generic", true}}
	gluPackedFuncs       = []gatedPackedKernelImpl{{GLUPackedF32Generic, "generic", true}}
	siGLUPackedFuncs     = []gatedPackedKernelImpl{{SiGLUPackedF32Generic, "generic", true}}
	seGLUPackedFuncs     = []gatedPackedKernelImpl{{SeGLUPackedF32Generic, "generic", true}}
	geGLUPackedFuncs     = []gatedPackedKernelImpl{{GeGLUPackedF32Generic, "generic", true}}
	geGLUTanhPackedFuncs = []gatedPackedKernelImpl{{GeGLUTanhPackedF32Generic, "generic", true}}
)
