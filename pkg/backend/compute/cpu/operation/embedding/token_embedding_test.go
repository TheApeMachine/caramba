package embedding

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestTokenEmbedding(t *testing.T) {
	Convey("Given a TokenEmbedding with vocab=4, d_model=8", t, func() {
		op := NewTokenEmbedding(4, 8, 0.02)

		Convey("Forward", func() {
			Convey("It should return the correct number of embedding vectors", func() {
				tokens := []float64{0, 1, 2, 3}
				out := op.Forward([]int{1, 4}, tokens)
				So(out, ShouldHaveLength, 4*8)
			})

			Convey("It should return the same vector for the same token id", func() {
				tokens := []float64{2, 2}
				out := op.Forward([]int{1, 2}, tokens)
				So(out[:8], ShouldResemble, out[8:])
			})

			Convey("It should return different vectors for different token ids", func() {
				tokens := []float64{0, 1}
				out := op.Forward([]int{1, 2}, tokens)
				same := true
				for i := 0; i < 8; i++ {
					if out[i] != out[8+i] {
						same = false
						break
					}
				}
				So(same, ShouldBeFalse)
			})
		})
	})
}

func BenchmarkTokenEmbedding_Forward(b *testing.B) {
	op := NewTokenEmbedding(32000, 512, 0.02)
	tokens := make([]float64, 512)
	for i := range tokens {
		tokens[i] = float64(i % 32000)
	}
	shape := []int{1, 512}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		op.Forward(shape, tokens)
	}
}
