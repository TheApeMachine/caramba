//go:build amd64

package activation

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func TestActivationAVX512DispatchParity(t *testing.T) {
	if !cpu.X86.HasAVX512F {
		t.Skip("AVX512F not supported")
	}

	convey.Convey("Given activation AVX-512 dispatch kernels", t, func() {
		type dispatchCase struct {
			name     string
			maxULP   int
			generic  func(dst, src *float32, count int)
			dispatch func(dst, src *float32, count int)
		}

		cases := []dispatchCase{
			{"ExpF32", 2, ExpF32Generic, expF32Kernel},
			{"LogF32", 2, LogF32Generic, logF32Kernel},
			{"Log1pF32", 2, Log1pF32Generic, log1pF32Kernel},
			{"Expm1F32", 2, Expm1F32Generic, expm1F32Kernel},
			{"SigmoidF32", 2, SigmoidF32Generic, sigmoidF32Kernel},
			{"LogSigmoidF32", 2, LogSigmoidF32Generic, logSigmoidF32Kernel},
			{"TanhF32", 2, TanhF32Generic, tanhF32Kernel},
			{"SiluF32", 2, SiluF32Generic, siluF32Kernel},
			{"GeluTanhF32", 2, GeluTanhF32Generic, geluTanhF32Kernel},
			{"GeluF32", 2, GeluF32Generic, geluF32Kernel},
			{"ReLUF32", 1, ReLUF32Generic, reluF32Kernel},
			{"LeakyReLUF32", 1, LeakyReLUF32Generic, leakyReluF32Kernel},
			{"ELUF32", 2, ELUF32Generic, eluF32Kernel},
			{"CELUF32", 2, CELUF32Generic, celuF32Kernel},
			{"SELUF32", 2, SELUF32Generic, seluF32Kernel},
			{"SoftplusF32", 2, SoftplusF32Generic, softplusF32Kernel},
			{"MishF32", 2, MishF32Generic, mishF32Kernel},
			{"SoftsignF32", 2, SoftsignF32Generic, softsignF32Kernel},
			{"HardSigmoidF32", 1, HardSigmoidF32Generic, hardSigmoidF32Kernel},
			{"HardSwishF32", 1, HardSwishF32Generic, hardSwishF32Kernel},
			{"HardTanhF32", 1, HardTanhF32Generic, hardTanhF32Kernel},
			{"HardGeluF32", 1, HardGeluF32Generic, hardGeluF32Kernel},
			{"QuickGeluF32", 1, QuickGeluF32Generic, quickGeluF32Kernel},
			{"TanhShrinkF32", 2, TanhShrinkF32Generic, tanhShrinkF32Kernel},
			{"SoftmaxF32", 2, SoftmaxF32Generic, softmaxF32Kernel},
			{"LogSoftmaxF32", 2, LogSoftmaxF32Generic, logSoftmaxF32Kernel},
		}

		for _, testCase := range cases {
			convey.Convey(testCase.name, func() {
				for _, count := range parity.Lengths {
					convey.Convey(fmt.Sprintf("N=%d", count), func() {
						source := make([]float32, count)
						want := make([]float32, count)
						got := make([]float32, count)

						for index := range source {
							source[index] = rand.Float32()*4.0 - 2.0
						}

						testCase.generic(&want[0], &source[0], count)
						testCase.dispatch(&got[0], &source[0], count)

						parity.AssertFloat32SlicesWithinULP(t, got, want, testCase.maxULP)
					})
				}
			})
		}
	})
}

func TestActivationAVX512ParamDispatchParity(t *testing.T) {
	if !cpu.X86.HasAVX512F {
		t.Skip("AVX512F not supported")
	}

	convey.Convey("Given parametric AVX-512 dispatch kernels", t, func() {
		slope := float32(0.1)
		threshold := float32(0.5)
		minVal := float32(-1.0)
		maxVal := float32(1.0)
		alpha := float32(1.5)
		lambda := float32(0.5)
		lower := float32(0.1)
		upper := float32(0.3)

		runParamSlopeParity(t, "LeakyReLUSlopeF32", 1, slope,
			LeakyReLUSlopeF32Generic, leakyReLUSlopeF32Kernel)
		runParamSlopeParity(t, "PReLUF32", 1, slope,
			PReLUF32Generic, preluF32Kernel)
		runParamSlopeParity(t, "ThresholdF32", 1, threshold,
			ThresholdF32Generic, thresholdF32Kernel)
		runParamRangeParity(t, "HardTanhRangeF32", 1, minVal, maxVal,
			HardTanhRangeF32Generic, hardTanhRangeF32Kernel)
		runParamSlopeParity(t, "ELUAlphaF32", 2, alpha,
			ELUAlphaF32Generic, eluAlphaF32Kernel)
		runParamSlopeParity(t, "CELUAlphaF32", 2, alpha,
			CELUAlphaF32Generic, celuAlphaF32Kernel)
		runParamSlopeParity(t, "HardShrinkF32", 1, lambda,
			HardShrinkF32Generic, hardShrinkF32Kernel)
		runParamSlopeParity(t, "SoftShrinkF32", 1, lambda,
			SoftShrinkF32Generic, softShrinkF32Kernel)
		runParamSlopeParity(t, "SnakeF32", 2, alpha,
			SnakeF32Generic, snakeF32Kernel)
		runParamRangeParity(t, "SnakeParametricF32", 2, alpha, slope,
			SnakeParametricF32Generic, snakeParametricF32Kernel)
		runParamRReluParity(t, "RReLUF32", 1, lower, upper,
			RReLUF32Generic, rreluF32Kernel)
		runParamIndexedParity(t, "PReLUVF32", 1,
			PReLUVF32Generic, preluVF32Kernel)
	})
}

func TestActivationAVX512GatedDispatchParity(t *testing.T) {
	if !cpu.X86.HasAVX512F {
		t.Skip("AVX512F not supported")
	}

	convey.Convey("Given gated AVX-512 dispatch kernels", t, func() {
		type gatedCase struct {
			name     string
			maxULP   int
			generic  func(dst, gate, up *float32, count int)
			dispatch func(dst, gate, up *float32, count int)
		}

		cases := []gatedCase{
			{"SwiGLUTensorsF32", 2, SwiGLUTensorsF32Generic, swiGLUTensorsKernel},
			{"LinGLUTensorsF32", 1, LinGLUTensorsF32Generic, linGLUTensorsKernel},
			{"ReGLUTensorsF32", 1, ReGLUTensorsF32Generic, reGLUTensorsKernel},
			{"GLUTensorsF32", 2, GLUTensorsF32Generic, gluTensorsKernel},
			{"SiGLUTensorsF32", 2, SiGLUTensorsF32Generic, siGLUTensorsKernel},
			{"SeGLUTensorsF32", 2, SeGLUTensorsF32Generic, seGLUTensorsKernel},
			{"GeGLUTensorsF32", 2, GeGLUTensorsF32Generic, geGLUTensorsKernel},
			{"GeGLUTanhTensorsF32", 2, GeGLUTanhTensorsF32Generic, geGLUTanhTensorsKernel},
		}

		for _, testCase := range cases {
			convey.Convey(testCase.name, func() {
				for _, count := range parity.Lengths {
					gate := make([]float32, count)
					up := make([]float32, count)
					want := make([]float32, count)
					got := make([]float32, count)

					for index := range gate {
						gate[index] = rand.Float32()*2.0 - 1.0
						up[index] = rand.Float32()*2.0 - 1.0
					}

					testCase.generic(&want[0], &gate[0], &up[0], count)
					testCase.dispatch(&got[0], &gate[0], &up[0], count)

					parity.AssertFloat32SlicesWithinULP(t, got, want, testCase.maxULP)
				}
			})
		}
	})
}

func BenchmarkActivationAVX512DispatchReLU(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX512F not supported")
	}

	count := 8192
	source := make([]float32, count)
	destination := make([]float32, count)

	for index := range source {
		source[index] = rand.Float32()*4.0 - 2.0
	}

	b.ResetTimer()
	for b.Loop() {
		reluF32Kernel(&destination[0], &source[0], count)
	}
}

func runParamSlopeParity(
	testingTB *testing.T,
	name string,
	maxULP int,
	slope float32,
	generic func(dst, src *float32, count int, slope float32),
	dispatch func(dst, src *float32, count int, slope float32),
) {
	convey.Convey(name, func() {
		for _, count := range parity.Lengths {
			source := make([]float32, count)
			want := make([]float32, count)
			got := make([]float32, count)

			for index := range source {
				source[index] = rand.Float32()*4.0 - 2.0
			}

			generic(&want[0], &source[0], count, slope)
			dispatch(&got[0], &source[0], count, slope)

			parity.AssertFloat32SlicesWithinULP(testingTB, got, want, maxULP)
		}
	})
}

func runParamRangeParity(
	testingTB *testing.T,
	name string,
	maxULP int,
	minVal, maxVal float32,
	generic func(dst, src *float32, count int, minVal, maxVal float32),
	dispatch func(dst, src *float32, count int, minVal, maxVal float32),
) {
	convey.Convey(name, func() {
		for _, count := range parity.Lengths {
			source := make([]float32, count)
			want := make([]float32, count)
			got := make([]float32, count)

			for index := range source {
				source[index] = rand.Float32()*4.0 - 2.0
			}

			generic(&want[0], &source[0], count, minVal, maxVal)
			dispatch(&got[0], &source[0], count, minVal, maxVal)

			parity.AssertFloat32SlicesWithinULP(testingTB, got, want, maxULP)
		}
	})
}

func runParamIndexedParity(
	testingTB *testing.T,
	name string,
	maxULP int,
	generic func(dst, src, slopes *float32, count int),
	dispatch func(dst, src, slopes *float32, count int),
) {
	convey.Convey(name, func() {
		for _, count := range parity.Lengths {
			source := make([]float32, count)
			slopes := make([]float32, count)
			want := make([]float32, count)
			got := make([]float32, count)

			for index := range source {
				source[index] = rand.Float32()*4.0 - 2.0
				slopes[index] = rand.Float32()*0.5 + 0.01
			}

			generic(&want[0], &source[0], &slopes[0], count)
			dispatch(&got[0], &source[0], &slopes[0], count)

			parity.AssertFloat32SlicesWithinULP(testingTB, got, want, maxULP)
		}
	})
}

func runParamRReluParity(
	testingTB *testing.T,
	name string,
	maxULP int,
	lower, upper float32,
	generic func(dst, src *float32, count int, lower, upper float32),
	dispatch func(dst, src *float32, count int, lower, upper float32),
) {
	convey.Convey(name, func() {
		for _, count := range parity.Lengths {
			source := make([]float32, count)
			want := make([]float32, count)
			got := make([]float32, count)

			for index := range source {
				source[index] = rand.Float32()*4.0 - 2.0
			}

			generic(&want[0], &source[0], count, lower, upper)
			dispatch(&got[0], &source[0], count, lower, upper)

			parity.AssertFloat32SlicesWithinULP(testingTB, got, want, maxULP)
		}
	})
}

func TestActivationAVX512GatedPackedDispatchParity(t *testing.T) {
	if !cpu.X86.HasAVX512F {
		t.Skip("AVX512F not supported")
	}

	convey.Convey("Given gated packed AVX-512 dispatch kernels", t, func() {
		type packedCase struct {
			name     string
			maxULP   int
			generic  func(dst, packed *float32, batch, halfCount int)
			dispatch func(dst, packed *float32, batch, halfCount int)
		}

		cases := []packedCase{
			{"SwiGLUPackedF32", 2, SwiGLUPackedF32Generic, swiGLUPackedKernel},
			{"LinGLUPackedF32", 1, LinGLUPackedF32Generic, linGLUPackedKernel},
			{"ReGLUPackedF32", 1, ReGLUPackedF32Generic, reGLUPackedKernel},
			{"GLUPackedF32", 2, GLUPackedF32Generic, gluPackedKernel},
			{"SiGLUPackedF32", 2, SiGLUPackedF32Generic, siGLUPackedKernel},
			{"SeGLUPackedF32", 2, SeGLUPackedF32Generic, seGLUPackedKernel},
			{"GeGLUPackedF32", 2, GeGLUPackedF32Generic, geGLUPackedKernel},
			{"GeGLUTanhPackedF32", 2, GeGLUTanhPackedF32Generic, geGLUTanhPackedKernel},
		}

		for _, testCase := range cases {
			convey.Convey(testCase.name, func() {
				for _, halfCount := range parity.Lengths {
					batch := 1
					packedLen := batch * halfCount * 2
					dstLen := batch * halfCount

					packed := make([]float32, packedLen)
					want := make([]float32, dstLen)
					got := make([]float32, dstLen)

					for index := range packed {
						packed[index] = rand.Float32()*2.0 - 1.0
					}

					testCase.generic(&want[0], &packed[0], batch, halfCount)
					testCase.dispatch(&got[0], &packed[0], batch, halfCount)

					parity.AssertFloat32SlicesWithinULP(t, got, want, testCase.maxULP)
				}
			})
		}
	})
}

func BenchmarkActivationAVX512DispatchSoftmax(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX512F not supported")
	}

	count := 8192
	source := make([]float32, count)
	destination := make([]float32, count)

	for index := range source {
		source[index] = rand.Float32()*2.0 - 1.0
	}

	b.ResetTimer()
	for b.Loop() {
		softmaxF32Kernel(&destination[0], &source[0], count)
	}
}
