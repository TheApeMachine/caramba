//go:build !amd64 && !arm64

package activation

var (
	expF32Funcs         = []f32KernelImpl{{ExpF32Generic, "generic", true}}
	logF32Funcs         = []f32KernelImpl{{LogF32Generic, "generic", true}}
	log1pF32Funcs       = []f32KernelImpl{{Log1pF32Generic, "generic", true}}
	expm1F32Funcs       = []f32KernelImpl{{Expm1F32Generic, "generic", true}}
	sigmoidF32Funcs     = []f32KernelImpl{{SigmoidF32Generic, "generic", true}}
	logSigmoidF32Funcs  = []f32KernelImpl{{LogSigmoidF32Generic, "generic", true}}
	tanhF32Funcs        = []f32KernelImpl{{TanhF32Generic, "generic", true}}
	siluF32Funcs        = []f32KernelImpl{{SiluF32Generic, "generic", true}}
	geluTanhF32Funcs    = []f32KernelImpl{{GeluTanhF32Generic, "generic", true}}
	geluF32Funcs        = []f32KernelImpl{{GeluF32Generic, "generic", true}}
	reluF32Funcs        = []f32KernelImpl{{ReLUF32Generic, "generic", true}}
	leakyReluF32Funcs   = []f32KernelImpl{{LeakyReLUF32Generic, "generic", true}}
	eluF32Funcs         = []f32KernelImpl{{ELUF32Generic, "generic", true}}
	celuF32Funcs        = []f32KernelImpl{{CELUF32Generic, "generic", true}}
	seluF32Funcs        = []f32KernelImpl{{SELUF32Generic, "generic", true}}
	softplusF32Funcs    = []f32KernelImpl{{SoftplusF32Generic, "generic", true}}
	mishF32Funcs        = []f32KernelImpl{{MishF32Generic, "generic", true}}
	softsignF32Funcs    = []f32KernelImpl{{SoftsignF32Generic, "generic", true}}
	hardSigmoidF32Funcs = []f32KernelImpl{{HardSigmoidF32Generic, "generic", true}}
	hardSwishF32Funcs   = []f32KernelImpl{{HardSwishF32Generic, "generic", true}}
	hardTanhF32Funcs    = []f32KernelImpl{{HardTanhF32Generic, "generic", true}}
	hardGeluF32Funcs    = []f32KernelImpl{{HardGeluF32Generic, "generic", true}}
	quickGeluF32Funcs   = []f32KernelImpl{{QuickGeluF32Generic, "generic", true}}
	tanhShrinkF32Funcs  = []f32KernelImpl{{TanhShrinkF32Generic, "generic", true}}
)
