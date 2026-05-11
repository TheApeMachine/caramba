package vsa_test

import (
	"math"
	"math/rand"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/vsa"
)

// randUnit returns a random unit-norm vector of length n.
func randUnit(n int, rng *rand.Rand) []float64 {
	v := make([]float64, n)
	sumsq := 0.0

	for i := range v {
		v[i] = rng.NormFloat64()
		sumsq += v[i] * v[i]
	}

	norm := math.Sqrt(sumsq)

	for i := range v {
		v[i] /= norm
	}

	return v
}

func TestBind(t *testing.T) {
	Convey("Given two VSA hypervectors", t, func() {
		n := 1024
		a := make([]float64, n)
		b := make([]float64, n)

		for i := range a {
			a[i] = 1.0 / math.Sqrt(float64(n))
			b[i] = 1.0 / math.Sqrt(float64(n))
		}

		Convey("It should bind them elementwise producing the Hadamard product", func() {
			op := vsa.NewBind()
			out := op.Forward([]int{n}, a, b)

			So(len(out), ShouldEqual, n)
			So(out[0], ShouldAlmostEqual, 1.0/float64(n), 1e-9)

			Convey("And the result length should equal n", func() {
				So(len(out), ShouldEqual, n)
			})
		})

		Convey("It should be commutative", func() {
			op := vsa.NewBind()
			ab := op.Forward([]int{n}, a, b)
			ba := op.Forward([]int{n}, b, a)

			for i := range ab {
				So(ab[i], ShouldAlmostEqual, ba[i], 1e-12)
			}
		})
	})
}

func TestBundle(t *testing.T) {
	Convey("Given two unit-norm VSA hypervectors", t, func() {
		n := 2048
		rng := rand.New(rand.NewSource(42))
		a := randUnit(n, rng)
		b := randUnit(n, rng)

		Convey("It should bundle them into a unit-norm result", func() {
			op := vsa.NewBundle()
			out := op.Forward([]int{n}, a, b)

			So(len(out), ShouldEqual, n)

			// Result must be unit norm
			sumsq := 0.0
			for _, v := range out {
				sumsq += v * v
			}

			So(math.Sqrt(sumsq), ShouldAlmostEqual, 1.0, 1e-9)
		})

		Convey("It should return a unit vector when given a single vector", func() {
			op := vsa.NewBundle()
			out := op.Forward([]int{n}, a)
			sumsq := 0.0

			for _, v := range out {
				sumsq += v * v
			}

			So(math.Sqrt(sumsq), ShouldAlmostEqual, 1.0, 1e-9)
		})
	})
}

func TestSimilarity(t *testing.T) {
	Convey("Given unit-norm VSA hypervectors", t, func() {
		n := 4096
		rng := rand.New(rand.NewSource(7))
		a := randUnit(n, rng)
		b := randUnit(n, rng)

		Convey("It should return 1.0 for a vector with itself", func() {
			op := vsa.NewSimilarity()
			out := op.Forward([]int{n}, a, a)

			So(len(out), ShouldEqual, 1)
			So(out[0], ShouldAlmostEqual, 1.0, 1e-9)
		})

		Convey("It should return a value close to 0 for two random orthogonal vectors", func() {
			op := vsa.NewSimilarity()
			out := op.Forward([]int{n}, a, b)

			// Expected cosine similarity ~ 0 for random high-dim vectors (within ~2/sqrt(n))
			So(math.Abs(out[0]), ShouldBeLessThan, 0.1)
		})

		Convey("It should return the same result symmetrically", func() {
			op := vsa.NewSimilarity()
			ab := op.Forward([]int{n}, a, b)
			ba := op.Forward([]int{n}, b, a)

			So(ab[0], ShouldAlmostEqual, ba[0], 1e-12)
		})
	})
}

func TestPermute(t *testing.T) {
	Convey("Given a VSA hypervector", t, func() {
		n := 512
		v := make([]float64, n)

		for i := range v {
			v[i] = float64(i)
		}

		Convey("It should shift elements cyclically by k positions", func() {
			k := 3
			op := vsa.NewPermute(k)
			out := op.Forward([]int{n}, v)

			So(len(out), ShouldEqual, n)
			So(out[k], ShouldEqual, v[0])
			So(out[k+1], ShouldEqual, v[1])
		})

		Convey("It should wrap correctly at the boundary", func() {
			k := n - 1
			op := vsa.NewPermute(k)
			out := op.Forward([]int{n}, v)

			So(out[n-1], ShouldEqual, v[0])
			So(out[0], ShouldEqual, v[1])
		})

		Convey("It should be a no-op when k is a multiple of n", func() {
			op := vsa.NewPermute(n)
			out := op.Forward([]int{n}, v)

			for i := range v {
				So(out[i], ShouldEqual, v[i])
			}
		})
	})
}

func TestInversePermute(t *testing.T) {
	Convey("Given a VSA hypervector", t, func() {
		n := 512
		v := make([]float64, n)

		for i := range v {
			v[i] = float64(i) * 1.5
		}

		Convey("It should recover the original vector after Permute + InversePermute", func() {
			for _, k := range []int{1, 7, 100, n - 1} {
				perm := vsa.NewPermute(k)
				inv := vsa.NewInversePermute(k)
				shifted := perm.Forward([]int{n}, v)
				recovered := inv.Forward([]int{n}, shifted)

				for i := range v {
					So(recovered[i], ShouldEqual, v[i])
				}
			}
		})

		Convey("It should handle negative-equivalent shifts correctly", func() {
			perm := vsa.NewPermute(5)
			inv := vsa.NewInversePermute(5)
			shifted := perm.Forward([]int{n}, v)
			recovered := inv.Forward([]int{n}, shifted)

			for i := range v {
				So(recovered[i], ShouldEqual, v[i])
			}
		})
	})
}

func BenchmarkBind(b *testing.B) {
	n := 10000
	rng := rand.New(rand.NewSource(1))
	a := randUnit(n, rng)
	v := randUnit(n, rng)
	op := vsa.NewBind()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		op.Forward([]int{n}, a, v)
	}
}

func BenchmarkBundle(b *testing.B) {
	n := 10000
	rng := rand.New(rand.NewSource(2))
	a := randUnit(n, rng)
	c := randUnit(n, rng)
	op := vsa.NewBundle()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		op.Forward([]int{n}, a, c)
	}
}

func BenchmarkSimilarity(b *testing.B) {
	n := 10000
	rng := rand.New(rand.NewSource(3))
	a := randUnit(n, rng)
	c := randUnit(n, rng)
	op := vsa.NewSimilarity()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		op.Forward([]int{n}, a, c)
	}
}

func BenchmarkPermute(b *testing.B) {
	n := 10000
	v := make([]float64, n)

	for i := range v {
		v[i] = float64(i)
	}

	op := vsa.NewPermute(42)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		op.Forward([]int{n}, v)
	}
}
