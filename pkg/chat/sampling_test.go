package chat

import (
	"math/rand"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewGenerationPolicy(test *testing.T) {
	Convey("Given model generation policy configuration", test, func() {
		Convey("It should reject invalid sampling controls", func() {
			_, err := newGenerationPolicy(ModelConfig{
				MaxNewTokens:      1,
				RepetitionPenalty: 1,
				Temperature:       -1,
			})

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "temperature")

			_, err = newGenerationPolicy(ModelConfig{
				MaxNewTokens:      1,
				RepetitionPenalty: 1,
				TopP:              1.5,
			})

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "top_p")

			_, err = newGenerationPolicy(ModelConfig{
				MaxNewTokens:      1,
				RepetitionPenalty: 1,
				TopK:              -1,
			})

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "top_k")
		})
	})
}

func TestGenerationPolicy_StopMatched(test *testing.T) {
	Convey("Given configured stop token sequences", test, func() {
		policy := generationPolicy{
			stopSequences: [][]int{{42}, {7, 8}},
		}

		Convey("It should detect completed stop suffixes", func() {
			So(policy.stopMatched([]int{1, 2, 42}), ShouldBeTrue)
			So(policy.stopMatched([]int{1, 7, 8}), ShouldBeTrue)
			So(policy.stopMatched([]int{1, 7}), ShouldBeFalse)
		})

		Convey("It should hold pending prefixes for multi-token stops", func() {
			So(policy.stopPending([]int{1, 7}), ShouldBeTrue)
			So(policy.stopPending([]int{1, 7, 8}), ShouldBeFalse)
		})
	})
}

func TestSelectLastTokenSampling(test *testing.T) {
	Convey("Given logits and a sampling policy", test, func() {
		Convey("It should keep greedy selection when temperature is zero", func() {
			tokenID, err := selectLastToken(
				[]int{1, 1, 4},
				[]float64{0, 10, 9, 1},
				nil,
				generationPolicy{temperature: 0, topP: 1},
			)

			So(err, ShouldBeNil)
			So(tokenID, ShouldEqual, 1)
		})

		Convey("It should respect top-k filtering while sampling", func() {
			tokenID, err := selectLastToken(
				[]int{1, 1, 4},
				[]float64{0, 10, 9, 1},
				nil,
				generationPolicy{
					temperature: 1,
					topK:        1,
					topP:        1,
					random:      rand.New(rand.NewSource(7)),
				},
			)

			So(err, ShouldBeNil)
			So(tokenID, ShouldEqual, 1)
		})
	})
}

func BenchmarkSelectLastToken(benchmark *testing.B) {
	values := make([]float64, 50257)
	policy := generationPolicy{
		repetitionPenalty: 1.1,
		temperature:       0.8,
		topK:              50,
		topP:              0.95,
		random:            rand.New(rand.NewSource(0)),
	}

	for index := range values {
		values[index] = float64(index%128) / 128
	}

	for benchmark.Loop() {
		_, _ = selectLastToken([]int{1, 1, len(values)}, values, []int{1, 2, 3}, policy)
	}
}
