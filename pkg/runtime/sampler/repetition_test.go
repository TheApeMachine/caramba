package sampler

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/program"
)

func TestRepetitionPenalty(t *testing.T) {
	Convey("Given a sampler with repetition_penalty 1.5", t, func() {
		runner := New(7)
		declaration := program.SamplerDeclaration{
			ID: "rep",
			Config: map[string]any{
				"temperature":        1.0,
				"top_k":              0,
				"repetition_penalty": 1.5,
			},
		}

		Convey("History tokens should be penalized away from a tied argmax", func() {
			logits := []float64{1.0, 1.0, 1.0, 1.0}
			counts := map[int]int{}

			for index := 0; index < 256; index++ {
				token, _, err := runner.Next(
					context.Background(), declaration, logits, []int{0, 1},
				)
				So(err, ShouldBeNil)
				counts[token]++
			}

			So(counts[2]+counts[3], ShouldBeGreaterThan, counts[0]+counts[1])
		})
	})

	Convey("Given repetition_penalty 1.0 the sampler should be a no-op", t, func() {
		runner := New(7)
		declaration := program.SamplerDeclaration{
			ID: "no-rep",
			Config: map[string]any{
				"temperature":        1.0,
				"repetition_penalty": 1.0,
			},
		}

		logits := []float64{1.0, 1.0, 1.0, 1.0}
		token, _, err := runner.Next(context.Background(), declaration, logits, []int{0, 1})
		So(err, ShouldBeNil)
		So(token, ShouldBeBetweenOrEqual, 0, 3)
	})

	Convey("Given repetition_penalty < 1 the config parser should reject it", t, func() {
		runner := New(7)
		declaration := program.SamplerDeclaration{
			ID: "bad",
			Config: map[string]any{
				"repetition_penalty": 0.5,
			},
		}

		_, _, err := runner.Next(context.Background(), declaration, []float64{1, 1}, nil)
		So(err, ShouldNotBeNil)
	})
}
