//go:build arm64

package activation

func ExpF32NEON(dst, src *float32, count int)
func LogF32NEON(dst, src *float32, count int)
func Log1pF32NEON(dst, src *float32, count int)
func Expm1F32NEON(dst, src *float32, count int)
func SigmoidF32NEON(dst, src *float32, count int)
func LogSigmoidF32NEON(dst, src *float32, count int)
func TanhF32NEON(dst, src *float32, count int)
func SiluF32NEON(dst, src *float32, count int)
func GeluTanhF32NEON(dst, src *float32, count int)
func GeluF32NEON(dst, src *float32, count int)
func ReLUF32NEON(dst, src *float32, count int)
func LeakyReLUF32NEON(dst, src *float32, count int)
func ELUF32NEON(dst, src *float32, count int)
func CELUF32NEON(dst, src *float32, count int)
func SELUF32NEON(dst, src *float32, count int)
func SoftplusF32NEON(dst, src *float32, count int)
func MishF32NEON(dst, src *float32, count int)
func SoftsignF32NEON(dst, src *float32, count int)
func HardSigmoidF32NEON(dst, src *float32, count int)
func HardSwishF32NEON(dst, src *float32, count int)
func HardTanhF32NEON(dst, src *float32, count int)
func HardGeluF32NEON(dst, src *float32, count int)
func QuickGeluF32NEON(dst, src *float32, count int)
func TanhShrinkF32NEON(dst, src *float32, count int)

var (
	expF32Funcs = []f32KernelImpl{
		{ExpF32NEON, "neon", true},
		{ExpF32Generic, "generic", true},
	}
	logF32Funcs = []f32KernelImpl{
		{LogF32NEON, "neon", true},
		{LogF32Generic, "generic", true},
	}
	log1pF32Funcs = []f32KernelImpl{
		{Log1pF32NEON, "neon", true},
		{Log1pF32Generic, "generic", true},
	}
	expm1F32Funcs = []f32KernelImpl{
		{Expm1F32NEON, "neon", true},
		{Expm1F32Generic, "generic", true},
	}
	sigmoidF32Funcs = []f32KernelImpl{
		{SigmoidF32NEON, "neon", true},
		{SigmoidF32Generic, "generic", true},
	}
	logSigmoidF32Funcs = []f32KernelImpl{
		{LogSigmoidF32NEON, "neon", true},
		{LogSigmoidF32Generic, "generic", true},
	}
	tanhF32Funcs = []f32KernelImpl{
		{TanhF32NEON, "neon", true},
		{TanhF32Generic, "generic", true},
	}
	siluF32Funcs = []f32KernelImpl{
		{SiluF32NEON, "neon", true},
		{SiluF32Generic, "generic", true},
	}
	geluTanhF32Funcs = []f32KernelImpl{
		{GeluTanhF32NEON, "neon", true},
		{GeluTanhF32Generic, "generic", true},
	}
	geluF32Funcs = []f32KernelImpl{
		{GeluF32NEON, "neon", true},
		{GeluF32Generic, "generic", true},
	}
	reluF32Funcs = []f32KernelImpl{
		{ReLUF32NEON, "neon", true},
		{ReLUF32Generic, "generic", true},
	}
	leakyReluF32Funcs = []f32KernelImpl{
		{LeakyReLUF32NEON, "neon", true},
		{LeakyReLUF32Generic, "generic", true},
	}
	eluF32Funcs = []f32KernelImpl{
		{ELUF32NEON, "neon", true},
		{ELUF32Generic, "generic", true},
	}
	celuF32Funcs = []f32KernelImpl{
		{CELUF32NEON, "neon", true},
		{CELUF32Generic, "generic", true},
	}
	seluF32Funcs = []f32KernelImpl{
		{SELUF32NEON, "neon", true},
		{SELUF32Generic, "generic", true},
	}
	softplusF32Funcs = []f32KernelImpl{
		{SoftplusF32NEON, "neon", true},
		{SoftplusF32Generic, "generic", true},
	}
	mishF32Funcs = []f32KernelImpl{
		{MishF32NEON, "neon", true},
		{MishF32Generic, "generic", true},
	}
	softsignF32Funcs = []f32KernelImpl{
		{SoftsignF32NEON, "neon", true},
		{SoftsignF32Generic, "generic", true},
	}
	hardSigmoidF32Funcs = []f32KernelImpl{
		{HardSigmoidF32NEON, "neon", true},
		{HardSigmoidF32Generic, "generic", true},
	}
	hardSwishF32Funcs = []f32KernelImpl{
		{HardSwishF32NEON, "neon", true},
		{HardSwishF32Generic, "generic", true},
	}
	hardTanhF32Funcs = []f32KernelImpl{
		{HardTanhF32NEON, "neon", true},
		{HardTanhF32Generic, "generic", true},
	}
	hardGeluF32Funcs = []f32KernelImpl{
		{HardGeluF32NEON, "neon", true},
		{HardGeluF32Generic, "generic", true},
	}
	quickGeluF32Funcs = []f32KernelImpl{
		{QuickGeluF32NEON, "neon", true},
		{QuickGeluF32Generic, "generic", true},
	}
	tanhShrinkF32Funcs = []f32KernelImpl{
		{TanhShrinkF32NEON, "neon", true},
		{TanhShrinkF32Generic, "generic", true},
	}
)
