//go:build arm64

package activation

func SwiGLUPackedF32NEON(dst, packed *float32, batch, halfCount int)
func LinGLUPackedF32NEON(dst, packed *float32, batch, halfCount int)
func ReGLUPackedF32NEON(dst, packed *float32, batch, halfCount int)
func GLUPackedF32NEON(dst, packed *float32, batch, halfCount int)
func SiGLUPackedF32NEON(dst, packed *float32, batch, halfCount int)
func SeGLUPackedF32NEON(dst, packed *float32, batch, halfCount int)
func GeGLUPackedF32NEON(dst, packed *float32, batch, halfCount int)
func GeGLUTanhPackedF32NEON(dst, packed *float32, batch, halfCount int)

var (
	swiGLUPackedFuncs = []gatedPackedKernelImpl{
		{SwiGLUPackedF32NEON, "neon", true},
		{SwiGLUPackedF32Generic, "generic", true},
	}
	linGLUPackedFuncs = []gatedPackedKernelImpl{
		{LinGLUPackedF32NEON, "neon", true},
		{LinGLUPackedF32Generic, "generic", true},
	}
	reGLUPackedFuncs = []gatedPackedKernelImpl{
		{ReGLUPackedF32NEON, "neon", true},
		{ReGLUPackedF32Generic, "generic", true},
	}
	gluPackedFuncs = []gatedPackedKernelImpl{
		{GLUPackedF32NEON, "neon", true},
		{GLUPackedF32Generic, "generic", true},
	}
	siGLUPackedFuncs = []gatedPackedKernelImpl{
		{SiGLUPackedF32NEON, "neon", true},
		{SiGLUPackedF32Generic, "generic", true},
	}
	seGLUPackedFuncs = []gatedPackedKernelImpl{
		{SeGLUPackedF32NEON, "neon", true},
		{SeGLUPackedF32Generic, "generic", true},
	}
	geGLUPackedFuncs = []gatedPackedKernelImpl{
		{GeGLUPackedF32NEON, "neon", true},
		{GeGLUPackedF32Generic, "generic", true},
	}
	geGLUTanhPackedFuncs = []gatedPackedKernelImpl{
		{GeGLUTanhPackedF32NEON, "neon", true},
		{GeGLUTanhPackedF32Generic, "generic", true},
	}
)
