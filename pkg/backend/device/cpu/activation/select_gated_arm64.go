//go:build arm64

package activation

func SwiGLUTensorsF32NEON(dst, gate, up *float32, count int)
func LinGLUTensorsF32NEON(dst, gate, up *float32, count int)
func ReGLUTensorsF32NEON(dst, gate, up *float32, count int)
func GLUTensorsF32NEON(dst, gate, up *float32, count int)
func SiGLUTensorsF32NEON(dst, gate, up *float32, count int)
func SeGLUTensorsF32NEON(dst, gate, up *float32, count int)
func GeGLUTensorsF32NEON(dst, gate, up *float32, count int)
func GeGLUTanhTensorsF32NEON(dst, gate, up *float32, count int)

var (
	swiGLUTensorsFuncs = []gatedTensorsKernelImpl{
		{SwiGLUTensorsF32NEON, "neon", true},
		{SwiGLUTensorsF32Generic, "generic", true},
	}
	linGLUTensorsFuncs = []gatedTensorsKernelImpl{
		{LinGLUTensorsF32NEON, "neon", true},
		{LinGLUTensorsF32Generic, "generic", true},
	}
	reGLUTensorsFuncs = []gatedTensorsKernelImpl{
		{ReGLUTensorsF32NEON, "neon", true},
		{ReGLUTensorsF32Generic, "generic", true},
	}
	gluTensorsFuncs = []gatedTensorsKernelImpl{
		{GLUTensorsF32NEON, "neon", true},
		{GLUTensorsF32Generic, "generic", true},
	}
	siGLUTensorsFuncs = []gatedTensorsKernelImpl{
		{SiGLUTensorsF32NEON, "neon", true},
		{SiGLUTensorsF32Generic, "generic", true},
	}
	seGLUTensorsFuncs = []gatedTensorsKernelImpl{
		{SeGLUTensorsF32NEON, "neon", true},
		{SeGLUTensorsF32Generic, "generic", true},
	}
	geGLUTensorsFuncs = []gatedTensorsKernelImpl{
		{GeGLUTensorsF32NEON, "neon", true},
		{GeGLUTensorsF32Generic, "generic", true},
	}
	geGLUTanhTensorsFuncs = []gatedTensorsKernelImpl{
		{GeGLUTanhTensorsF32NEON, "neon", true},
		{GeGLUTanhTensorsF32Generic, "generic", true},
	}
)
