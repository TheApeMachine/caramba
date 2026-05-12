package markov_blanket

import (
	"fmt"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestPartition(t *testing.T) {
	Convey("Given a Partition operation", t, func() {
		op := NewPartition()

		Convey("Forward", func() {
			Convey("It should extract sensory states correctly", func() {
				// N=4: x=[10,20,30,40], sensory={0,2}, active={1}, internal={3}, external={}
				x := []float64{10, 20, 30, 40}
				smask := []float64{1, 0, 1, 0}
				amask := []float64{0, 1, 0, 0}
				imask := []float64{0, 0, 0, 1}
				emask := []float64{0, 0, 0, 0}
				shape := []int{4, 2, 1, 1, 0}
				out := op.Forward(shape, x, smask, amask, imask, emask)
				// Sensory: [10, 30], Active: [20], Internal: [40]
				So(out, ShouldHaveLength, 4)
				So(out[0], ShouldEqual, 10)
				So(out[1], ShouldEqual, 30)
				So(out[2], ShouldEqual, 20)
				So(out[3], ShouldEqual, 40)
			})

			Convey("It should panic on shape length < 5", func() {
				x := []float64{10, 20, 30, 40}
				smask := []float64{1, 0, 1, 0}
				amask := []float64{0, 1, 0, 0}
				imask := []float64{0, 0, 0, 1}
				emask := []float64{0, 0, 0, 0}
				So(func() { op.Forward([]int{4, 2}, x, smask, amask, imask, emask) }, ShouldPanic)
			})

			Convey("It should panic on data length < 5", func() {
				shape := []int{4, 2, 1, 1, 0}
				x := []float64{1, 2, 3, 4}
				smask := []float64{1, 0, 1, 0}
				amask := []float64{0, 1, 0, 0}
				imask := []float64{0, 0, 0, 1}
				So(func() {
					op.Forward(shape, x, smask, amask, imask)
				}, ShouldPanic)
			})
		})
	})
}

func TestFlowInternal(t *testing.T) {
	Convey("Given a FlowInternal operation", t, func() {
		op := NewFlowInternal()

		Convey("Forward", func() {
			Convey("It should compute W @ x_sens + bias correctly", func() {
				// Ni=2, Ns=3
				// W = [[1,0,0],[0,1,0]]  x_sens=[5,7,9]  bias=[1,2]
				// out = [5+1, 7+2] = [6, 9]
				xSens := []float64{5, 7, 9}
				w := []float64{1, 0, 0, 0, 1, 0}
				bias := []float64{1, 2}
				out := op.Forward([]int{2, 3}, xSens, w, bias)
				So(out, ShouldHaveLength, 2)
				So(out[0], ShouldAlmostEqual, 6.0, 1e-9)
				So(out[1], ShouldAlmostEqual, 9.0, 1e-9)
			})

			Convey("It should produce zero output for zero weight matrix", func() {
				xSens := []float64{1, 2, 3}
				w := make([]float64, 4*3)
				bias := make([]float64, 4)
				out := op.Forward([]int{4, 3}, xSens, w, bias)
				for _, val := range out {
					So(val, ShouldAlmostEqual, 0.0, 1e-12)
				}
			})

			Convey("It should panic on shape length < 2", func() {
				So(func() {
					defer func() {
						recovered := recover()
						So(recovered, ShouldNotBeNil)
						So(fmt.Sprint(recovered), ShouldContainSubstring, "need >= 2")
					}()
					op.Forward([]int{4}, nil, nil, nil)
				}, ShouldNotPanic)
			})

			Convey("It should panic when shape[2] != N_i", func() {
				xSens := []float64{1, 2, 3}
				w := make([]float64, 2*3)
				bias := []float64{0, 0}
				So(func() {
					op.Forward([]int{2, 3, 99}, xSens, w, bias)
				}, ShouldPanic)
			})
		})
	})
}

func TestFlowActive(t *testing.T) {
	Convey("Given a FlowActive operation", t, func() {
		op := NewFlowActive()

		Convey("Forward", func() {
			Convey("It should compute W @ x_int + bias correctly", func() {
				// Na=2, Ni=2
				// W = [[2,0],[0,3]]  x_int=[4,5]  bias=[0,0]
				// out = [8, 15]
				xInt := []float64{4, 5}
				w := []float64{2, 0, 0, 3}
				bias := []float64{0, 0}
				out := op.Forward([]int{2, 2}, xInt, w, bias)
				So(out, ShouldHaveLength, 2)
				So(out[0], ShouldAlmostEqual, 8.0, 1e-9)
				So(out[1], ShouldAlmostEqual, 15.0, 1e-9)
			})

			Convey("It should incorporate bias correctly", func() {
				xInt := []float64{1, 0}
				w := []float64{1, 0, 0, 1}
				bias := []float64{10, 20}
				out := op.Forward([]int{2, 2}, xInt, w, bias)
				So(out[0], ShouldAlmostEqual, 11.0, 1e-9)
				So(out[1], ShouldAlmostEqual, 20.0, 1e-9)
			})

			Convey("It should panic on shape length < 2", func() {
				So(func() {
					defer func() {
						recovered := recover()
						So(recovered, ShouldNotBeNil)
						So(fmt.Sprint(recovered), ShouldContainSubstring, "need >= 2")
					}()
					op.Forward([]int{4}, nil, nil, nil)
				}, ShouldNotPanic)
			})
		})
	})
}

func TestMutualInformation(t *testing.T) {
	Convey("Given a MutualInformation operation", t, func() {
		op := NewMutualInformation()

		Convey("Forward", func() {
			Convey("It should return non-negative MI for correlated signals", func() {
				// T=100 samples, N=1, M=1
				// X = Y (perfect correlation) → high MI
				T := 100
				x := make([]float64, T)
				y := make([]float64, T)

				for idx := range T {
					val := float64(idx) / float64(T)
					x[idx] = val
					y[idx] = val
				}

				out := op.Forward([]int{1, 1}, x, y)
				So(out, ShouldHaveLength, 1)
				So(out[0], ShouldBeGreaterThanOrEqualTo, 0.0)
			})

			Convey("It should return near-zero MI for independent signals", func() {
				// X and Y drawn from independent constant distributions
				T := 50
				x := make([]float64, T)
				y := make([]float64, T)

				for idx := range T {
					x[idx] = 1.0
					y[idx] = -1.0
				}

				out := op.Forward([]int{1, 1}, x, y)
				So(out, ShouldHaveLength, 1)
				So(out[0], ShouldAlmostEqual, 0.0, 1e-6)
			})

			Convey("It should panic on insufficient shape", func() {
				x := []float64{1.0, 2.0}
				y := []float64{3.0, 4.0}
				So(func() { op.Forward([]int{2}, x, y) }, ShouldPanic)
			})
		})
	})
}

func BenchmarkFlowInternal_Forward(b *testing.B) {
	op := NewFlowInternal()
	Ni, Ns := 256, 256
	xSens := make([]float64, Ns)
	w := make([]float64, Ni*Ns)
	bias := make([]float64, Ni)

	for idx := range Ns {
		xSens[idx] = float64(idx) / float64(Ns)
	}

	for idx := range Ni * Ns {
		w[idx] = float64(idx) / float64(Ni*Ns)
	}

	for idx := range Ni {
		bias[idx] = 0.1
	}

	shape := []int{Ni, Ns}
	b.ResetTimer()

	for range b.N {
		op.Forward(shape, xSens, w, bias)
	}
}

func BenchmarkFlowActive_Forward(b *testing.B) {
	op := NewFlowActive()
	Na, Ni := 256, 256
	xInt := make([]float64, Ni)
	w := make([]float64, Na*Ni)
	bias := make([]float64, Na)

	for idx := range Ni {
		xInt[idx] = float64(idx) / float64(Ni)
	}

	for idx := range Na {
		bias[idx] = 0.1
	}

	shape := []int{Na, Ni}
	b.ResetTimer()

	for range b.N {
		op.Forward(shape, xInt, w, bias)
	}
}

func BenchmarkMutualInformation_Forward(b *testing.B) {
	op := NewMutualInformation()
	T, N, M := 1000, 8, 8
	x := make([]float64, T*N)
	y := make([]float64, T*M)

	for idx := range T * N {
		x[idx] = float64(idx) / float64(T*N)
	}

	for idx := range T * M {
		y[idx] = float64(idx) / float64(T*M)
	}

	shape := []int{N, M}
	b.ResetTimer()

	for range b.N {
		op.Forward(shape, x, y)
	}
}
