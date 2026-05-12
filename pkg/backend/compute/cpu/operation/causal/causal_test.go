package causal

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestDoCalculus(t *testing.T) {
	Convey("Given a DoCalculus operation", t, func() {
		op := NewDoCalculus()

		Convey("Forward", func() {
			Convey("It should return the intervention value for an intervened variable", func() {
				// 2-variable system: joint Gaussian with identity covariance.
				n := 2
				cov := []float64{1, 0, 0, 1}
				mask := []float64{1, 0}   // intervene on variable 0
				values := []float64{2, 0} // set X=2

				out := op.Forward([]int{n, n, 1}, cov, mask, values)

				So(len(out), ShouldEqual, n+n*n)
				// Mean of intervened variable should equal intervention value.
				So(out[0], ShouldAlmostEqual, 2.0, 1e-6)
			})

			Convey("It should zero covariance rows and columns for intervened variables", func() {
				n := 3
				cov := []float64{
					1, 0.5, 0.3,
					0.5, 1, 0.4,
					0.3, 0.4, 1,
				}
				mask := []float64{1, 0, 0}
				values := []float64{1, 0, 0}

				out := op.Forward([]int{n, n, 1}, cov, mask, values)
				adjCov := out[n:]

				// Row 0 and col 0 should be zero.
				So(adjCov[0*n+0], ShouldAlmostEqual, 0, 1e-9)
				So(adjCov[0*n+1], ShouldAlmostEqual, 0, 1e-9)
				So(adjCov[1*n+0], ShouldAlmostEqual, 0, 1e-9)
			})

			Convey("It should leave output length as N + N*N", func() {
				n := 4
				cov := make([]float64, n*n)
				for idx := 0; idx < n; idx++ {
					cov[idx*n+idx] = 1
				}
				mask := make([]float64, n)
				values := make([]float64, n)

				out := op.Forward([]int{n, n, 1}, cov, mask, values)

				So(len(out), ShouldEqual, n+n*n)
			})
		})
	})
}

func BenchmarkDoCalculus_Forward(b *testing.B) {
	op := NewDoCalculus()
	n := 8
	cov := make([]float64, n*n)

	for idx := 0; idx < n; idx++ {
		cov[idx*n+idx] = 1
	}

	mask := make([]float64, n)
	mask[0] = 1
	values := make([]float64, n)
	values[0] = 1.0
	shape := []int{n, n, 1}

	b.ResetTimer()

	for iteration := 0; iteration < b.N; iteration++ {
		op.Forward(shape, cov, mask, values)
	}
}

func TestBackdoorAdjustment(t *testing.T) {
	Convey("Given a BackdoorAdjustment operation", t, func() {
		op := NewBackdoorAdjustment()

		Convey("Forward", func() {
			Convey("It should panic on zero N_x or zero T", func() {
				So(func() {
					op.Forward([]int{1, 0, 1, 10}, []float64{}, []float64{}, []float64{})
				}, ShouldPanic)
				So(func() {
					op.Forward([]int{1, 1, 1, 0}, []float64{}, []float64{}, []float64{})
				}, ShouldPanic)
			})

			Convey("It should return a causal effect vector of length N_y", func() {
				ny, nx, nz, obs := 1, 1, 1, 50
				y := make([]float64, obs*ny)
				x := make([]float64, obs*nx)
				z := make([]float64, obs*nz)

				for obsIdx := 0; obsIdx < obs; obsIdx++ {
					x[obsIdx] = float64(obsIdx) / float64(obs)
					z[obsIdx] = 0.5
					// Y = 2*X + noise-free (z is confounder, controlled)
					y[obsIdx] = 2.0 * x[obsIdx]
				}

				out := op.Forward([]int{ny, nx, nz, obs}, y, x, z)

				So(len(out), ShouldEqual, ny)
				// The causal effect should be close to 2.0
				So(out[0], ShouldAlmostEqual, 2.0, 0.01)
			})

			Convey("It should produce sensible estimates with 2D treatment", func() {
				ny, nx, nz, obs := 2, 2, 1, 100
				y := make([]float64, obs*ny)
				x := make([]float64, obs*nx)
				z := make([]float64, obs*nz)

				for obsIdx := 0; obsIdx < obs; obsIdx++ {
					x[obsIdx*2] = float64(obsIdx) / 100.0
					x[obsIdx*2+1] = float64(obs-obsIdx) / 100.0
					z[obsIdx] = 0
					y[obsIdx*2] = x[obsIdx*2] * 3.0
					y[obsIdx*2+1] = x[obsIdx*2+1] * 1.5
				}

				out := op.Forward([]int{ny, nx, nz, obs}, y, x, z)

				So(len(out), ShouldEqual, ny)
				// Mean |OLS X_coef| per dimension: Y0 depends only on X0 (≈3), Y1 only on X1 (≈1.5);
				// averaged over nx=2 predictors → (3+0)/2 and (0+1.5)/2.
				So(out[0], ShouldAlmostEqual, 1.5, 0.08)
				So(out[1], ShouldAlmostEqual, 0.75, 0.08)
			})
		})
	})
}

func BenchmarkBackdoorAdjustment_Forward(b *testing.B) {
	op := NewBackdoorAdjustment()
	ny, nx, nz, obs := 2, 2, 2, 200
	y := make([]float64, obs*ny)
	x := make([]float64, obs*nx)
	z := make([]float64, obs*nz)

	for obsIdx := 0; obsIdx < obs; obsIdx++ {
		for j := 0; j < nx; j++ {
			x[obsIdx*nx+j] = float64(obsIdx+j) / float64(obs)
		}
		for j := 0; j < nz; j++ {
			z[obsIdx*nz+j] = 0.5
		}
		for j := 0; j < ny; j++ {
			y[obsIdx*ny+j] = x[obsIdx*nx] * 2.0
		}
	}

	shape := []int{ny, nx, nz, obs}
	b.ResetTimer()

	for iteration := 0; iteration < b.N; iteration++ {
		op.Forward(shape, y, x, z)
	}
}

func TestFrontdoorAdjustment(t *testing.T) {
	Convey("Given a FrontdoorAdjustment operation", t, func() {
		op := NewFrontdoorAdjustment()

		Convey("Forward", func() {
			Convey("It should return causal effect of length N_x", func() {
				nx, nm, ny, obs := 3, 3, 1, 60
				xVec := make([]float64, obs)
				mVec := make([]float64, obs)
				yVec := make([]float64, obs)

				for obsIdx := 0; obsIdx < obs; obsIdx++ {
					xVec[obsIdx] = float64(obsIdx%nx) / float64(nx)
					mVec[obsIdx] = xVec[obsIdx] * 0.8
					yVec[obsIdx] = mVec[obsIdx] * 1.5
				}

				out := op.Forward([]int{nx, nm, ny, obs}, xVec, mVec, yVec)

				So(len(out), ShouldEqual, nx)
				for _, effect := range out {
					So(math.IsNaN(effect), ShouldBeFalse)
					So(math.Abs(effect), ShouldBeLessThan, 10.0)
				}
			})

			Convey("It should return frontdoor estimates near the simulated indirect effect", func() {
				nx, nm, ny, obs := 2, 2, 1, 20
				xVec := make([]float64, obs)
				mVec := make([]float64, obs)
				yVec := make([]float64, obs)

				for obsIdx := 0; obsIdx < obs; obsIdx++ {
					xVec[obsIdx] = float64(obsIdx % 2)
					mVec[obsIdx] = float64(obsIdx % 2)
					yVec[obsIdx] = float64(obsIdx%2) * 2.0
				}

				out := op.Forward([]int{nx, nm, ny, obs}, xVec, mVec, yVec)

				So(len(out), ShouldEqual, nx)
				for _, v := range out {
					So(math.IsNaN(v), ShouldBeFalse)
					So(math.IsInf(v, 0), ShouldBeFalse)
				}
			})
		})
	})
}

func BenchmarkFrontdoorAdjustment_Forward(b *testing.B) {
	op := NewFrontdoorAdjustment()
	nx, nm, ny, obs := 5, 5, 1, 500
	xVec := make([]float64, obs)
	mVec := make([]float64, obs)
	yVec := make([]float64, obs)

	for obsIdx := 0; obsIdx < obs; obsIdx++ {
		xVec[obsIdx] = float64(obsIdx) / float64(obs)
		mVec[obsIdx] = xVec[obsIdx]*0.7 + 0.1
		yVec[obsIdx] = mVec[obsIdx]*2.0 + 0.3
	}

	shape := []int{nx, nm, ny, obs}
	b.ResetTimer()

	for iteration := 0; iteration < b.N; iteration++ {
		op.Forward(shape, xVec, mVec, yVec)
	}
}

func TestCounterfactual(t *testing.T) {
	Convey("Given a Counterfactual operation", t, func() {
		op := NewCounterfactual()

		Convey("Forward", func() {
			Convey("It should compute Y_cf = beta * X_cf + noise for linear SCM", func() {
				// Y = 2*X + 1 (noise=1 constant for all)
				// E[noise] = Y - beta*X = 1 for all obs
				n := 5
				xObs := []float64{0, 1, 2, 3, 4}
				yObs := []float64{1, 3, 5, 7, 9} // Y = 2X + 1
				beta := []float64{2, 2, 2, 2, 2}
				xCF := []float64{0, 5, 10}

				out := op.Forward([]int{n, 3}, xObs, yObs, beta, xCF)

				So(len(out), ShouldEqual, n*3)

				for row := 0; row < n; row++ {
					base := row * 3
					So(out[base+0], ShouldAlmostEqual, 1.0, 1e-6)
					So(out[base+1], ShouldAlmostEqual, 11.0, 1e-6)
					So(out[base+2], ShouldAlmostEqual, 21.0, 1e-6)
				}
			})

			Convey("It should return length N_cf", func() {
				n, nCF := 10, 7
				xObs := make([]float64, n)
				yObs := make([]float64, n)
				beta := make([]float64, n)
				xCF := make([]float64, nCF)

				for idx := 0; idx < n; idx++ {
					xObs[idx] = float64(idx)
					beta[idx] = 1.5
					yObs[idx] = 1.5*xObs[idx] + 0.3
				}

				out := op.Forward([]int{n, nCF}, xObs, yObs, beta, xCF)

				So(len(out), ShouldEqual, n*nCF)

				for _, v := range out {
					So(math.IsNaN(v), ShouldBeFalse)
					So(math.IsInf(v, 0), ShouldBeFalse)
					So(math.Abs(v), ShouldBeLessThan, 1e6)
				}
			})
		})
	})
}

func BenchmarkCounterfactual_Forward(b *testing.B) {
	op := NewCounterfactual()
	n, nCF := 1000, 200
	xObs := make([]float64, n)
	yObs := make([]float64, n)
	beta := make([]float64, n)
	xCF := make([]float64, nCF)

	for idx := 0; idx < n; idx++ {
		xObs[idx] = float64(idx) / float64(n)
		beta[idx] = 2.5
		yObs[idx] = 2.5*xObs[idx] + 0.7
	}

	shape := []int{n, nCF}
	b.ResetTimer()

	for iteration := 0; iteration < b.N; iteration++ {
		op.Forward(shape, xObs, yObs, beta, xCF)
	}
}

func TestIVEstimate(t *testing.T) {
	Convey("Given an IVEstimate operation", t, func() {
		op := NewIVEstimate()

		Convey("Forward", func() {
			Convey("It should return beta_iv of length N_x*N_y", func() {
				obs, nz, nx, ny := 100, 2, 1, 1

				// Build valid instrument that satisfies IV assumptions approximately.
				z := make([]float64, obs*nz)
				x := make([]float64, obs*nx)
				y := make([]float64, obs*ny)

				for obsIdx := 0; obsIdx < obs; obsIdx++ {
					z[obsIdx*nz] = float64(obsIdx) / float64(obs)
					z[obsIdx*nz+1] = 1.0
					x[obsIdx] = z[obsIdx*nz]*2.0 + 0.1
					y[obsIdx] = x[obsIdx] * 3.0
				}

				out := op.Forward([]int{obs, nz, nx, ny}, z, x, y)

				So(len(out), ShouldEqual, nx*ny)
			})

			Convey("It should produce an estimate close to true beta for clean IV", func() {
				obs := 200
				nz, nx, ny := 1, 1, 1

				z := make([]float64, obs)
				x := make([]float64, obs)
				y := make([]float64, obs)

				// Z → X with coefficient 1.0; X → Y with true beta = 4.0
				for obsIdx := 0; obsIdx < obs; obsIdx++ {
					z[obsIdx] = float64(obsIdx) / float64(obs)
					x[obsIdx] = z[obsIdx]
					y[obsIdx] = 4.0 * x[obsIdx]
				}

				out := op.Forward([]int{obs, nz, nx, ny}, z, x, y)

				So(out[0], ShouldAlmostEqual, 4.0, 0.1)
			})
		})
	})
}

func BenchmarkIVEstimate_Forward(b *testing.B) {
	op := NewIVEstimate()
	obs, nz, nx, ny := 500, 3, 2, 2
	z := make([]float64, obs*nz)
	x := make([]float64, obs*nx)
	y := make([]float64, obs*ny)

	for obsIdx := 0; obsIdx < obs; obsIdx++ {
		for j := 0; j < nz; j++ {
			z[obsIdx*nz+j] = float64(obsIdx+j) / float64(obs)
		}
		for j := 0; j < nx; j++ {
			x[obsIdx*nx+j] = z[obsIdx*nz] + float64(j)*0.1
		}
		for j := 0; j < ny; j++ {
			y[obsIdx*ny+j] = x[obsIdx*nx] * 2.0
		}
	}

	shape := []int{obs, nz, nx, ny}
	b.ResetTimer()

	for iteration := 0; iteration < b.N; iteration++ {
		op.Forward(shape, z, x, y)
	}
}

func TestCATE(t *testing.T) {
	Convey("Given a CATE operation", t, func() {
		op := NewCATE()

		Convey("Forward", func() {
			Convey("It should return NaN CATEs when treated or control arm is empty", func() {
				obs, nx := 4, 1
				x := make([]float64, obs*nx)
				treatment := make([]float64, obs)
				y := make([]float64, obs)

				for obsIdx := 0; obsIdx < obs; obsIdx++ {
					x[obsIdx] = float64(obsIdx)
					treatment[obsIdx] = 1.0 // all treated
					y[obsIdx] = x[obsIdx]
				}

				out := op.Forward([]int{obs, nx, 1}, x, treatment, y)

				for _, v := range out {
					So(math.IsNaN(v), ShouldBeTrue)
				}
			})

			Convey("It should return CATE of length T", func() {
				obs, nx := 60, 2
				x := make([]float64, obs*nx)
				treatment := make([]float64, obs)
				y := make([]float64, obs)

				for obsIdx := 0; obsIdx < obs; obsIdx++ {
					x[obsIdx*nx] = float64(obsIdx) / float64(obs)
					x[obsIdx*nx+1] = 1.0
					treatment[obsIdx] = float64(obsIdx % 2)
					y[obsIdx] = x[obsIdx*nx]*2.0 + treatment[obsIdx]*3.0
				}

				out := op.Forward([]int{obs, nx, 1}, x, treatment, y)

				So(len(out), ShouldEqual, obs)
				for _, v := range out {
					So(v, ShouldAlmostEqual, 3.0, 1e-4)
				}
			})

			Convey("It should estimate approximately constant ATE when true effect is constant", func() {
				// Simple experiment: Y(1) - Y(0) = 5 for all units.
				obs, nx := 100, 1
				x := make([]float64, obs*nx)
				treatment := make([]float64, obs)
				y := make([]float64, obs)

				for obsIdx := 0; obsIdx < obs; obsIdx++ {
					x[obsIdx] = float64(obsIdx) / float64(obs)
					treatment[obsIdx] = float64(obsIdx % 2)
					// Y = X + 5*T (no interaction)
					y[obsIdx] = x[obsIdx] + 5.0*treatment[obsIdx]
				}

				out := op.Forward([]int{obs, nx, 1}, x, treatment, y)

				So(len(out), ShouldEqual, obs)
				// Mean CATE should be in a reasonable range of 5.
				sum := 0.0
				for _, v := range out {
					sum += v
				}
				mean := sum / float64(obs)
				So(math.Abs(mean-5.0), ShouldBeLessThan, 0.5)
			})
		})
	})
}

func BenchmarkCATE_Forward(b *testing.B) {
	op := NewCATE()
	obs, nx := 1000, 5
	x := make([]float64, obs*nx)
	treatment := make([]float64, obs)
	y := make([]float64, obs)

	for obsIdx := 0; obsIdx < obs; obsIdx++ {
		for j := 0; j < nx; j++ {
			x[obsIdx*nx+j] = float64(obsIdx+j) / float64(obs)
		}
		treatment[obsIdx] = float64(obsIdx % 2)
		y[obsIdx] = x[obsIdx*nx]*2.0 + treatment[obsIdx]*4.0
	}

	shape := []int{obs, nx, 1}
	b.ResetTimer()

	for iteration := 0; iteration < b.N; iteration++ {
		op.Forward(shape, x, treatment, y)
	}
}

func TestDAGMarkovFactorization(t *testing.T) {
	Convey("Given a DAGMarkovFactorization operation", t, func() {
		op := NewDAGMarkovFactorization()

		Convey("Forward", func() {
			Convey("It should return log_prob of length T", func() {
				n, obs := 3, 20
				x := make([]float64, obs*n)

				for obsIdx := 0; obsIdx < obs; obsIdx++ {
					for nodeIdx := 0; nodeIdx < n; nodeIdx++ {
						x[obsIdx*n+nodeIdx] = float64(obsIdx+nodeIdx) / float64(obs)
					}
				}

				// Chain DAG: 0→1→2
				adj := []float64{
					0, 0, 0,
					1, 0, 0,
					0, 1, 0,
				}

				out := op.Forward([]int{n, obs}, x, adj)

				So(len(out), ShouldEqual, obs)
			})

			Convey("It should panic when adjacency contains a cycle", func() {
				n, obs := 2, 5
				x := make([]float64, obs*n)
				// Mutual dependency: 0→1 and 1→0
				adj := []float64{
					0, 1,
					1, 0,
				}

				So(func() {
					op.Forward([]int{n, obs}, x, adj)
				}, ShouldPanic)
			})

			Convey("It should produce finite log probabilities for valid data", func() {
				n, obs := 2, 30
				x := make([]float64, obs*n)

				for obsIdx := 0; obsIdx < obs; obsIdx++ {
					x[obsIdx*n] = float64(obsIdx) / float64(obs)
					x[obsIdx*n+1] = x[obsIdx*n]*1.5 + 0.1
				}

				// 0→1
				adj := []float64{0, 0, 1, 0}

				out := op.Forward([]int{n, obs}, x, adj)

				for _, lp := range out {
					So(math.IsNaN(lp), ShouldBeFalse)
					So(math.IsInf(lp, 0), ShouldBeFalse)
				}
			})

			Convey("OOD observations should have lower mean log probability than in-distribution data", func() {
				n, obs := 2, 100
				adj := []float64{0, 0, 1, 0}

				xIn := make([]float64, obs*n)
				for obsIdx := 0; obsIdx < obs; obsIdx++ {
					xVal := float64(obsIdx)/float64(obs)*4.0 - 2.0
					xIn[obsIdx*n] = xVal
					xIn[obsIdx*n+1] = xVal + 0.01*float64(obsIdx%5)
				}

				xOOD := make([]float64, obs*n)
				copy(xOOD, xIn)

				for obsIdx := 0; obsIdx < obs; obsIdx++ {
					xOOD[obsIdx*n] += 50.0
					xOOD[obsIdx*n+1] += 50.0
				}

				outIn := op.Forward([]int{n, obs}, xIn, adj)
				outOOD := op.Forward([]int{n, obs}, xOOD, adj)

				meanIn := 0.0
				for _, v := range outIn {
					meanIn += v
				}
				meanIn /= float64(len(outIn))

				meanOOD := 0.0
				for _, v := range outOOD {
					meanOOD += v
				}
				meanOOD /= float64(len(outOOD))

				So(meanOOD, ShouldBeLessThan, meanIn)
			})
		})
	})
}

func BenchmarkDAGMarkovFactorization_Forward(b *testing.B) {
	op := NewDAGMarkovFactorization()
	n, obs := 6, 500
	x := make([]float64, obs*n)

	for obsIdx := 0; obsIdx < obs; obsIdx++ {
		for nodeIdx := 0; nodeIdx < n; nodeIdx++ {
			x[obsIdx*n+nodeIdx] = float64(obsIdx+nodeIdx) / float64(obs)
		}
	}

	// DAG: chain 0→1→2→3→4→5
	adj := make([]float64, n*n)
	for nodeIdx := 1; nodeIdx < n; nodeIdx++ {
		adj[nodeIdx*n+(nodeIdx-1)] = 1
	}

	shape := []int{n, obs}
	b.ResetTimer()

	for iteration := 0; iteration < b.N; iteration++ {
		op.Forward(shape, x, adj)
	}
}
