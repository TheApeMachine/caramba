package sampler

import (
	"context"
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/program"
)

func TestCategoricalNext(t *testing.T) {
	Convey("Given a categorical sampler with two equally weighted logits", t, func() {
		runner := New(42)
		declaration := program.SamplerDeclaration{
			ID:     "main",
			Type:   "categorical",
			Config: map[string]any{"temperature": 1.0},
		}

		Convey("Repeated draws should produce both tokens", func() {
			seen := map[int]bool{}

			for index := 0; index < 64; index++ {
				token, _, err := runner.Next(context.Background(), declaration, []float64{1.0, 1.0}, nil)
				So(err, ShouldBeNil)
				seen[token] = true
			}

			So(seen[0], ShouldBeTrue)
			So(seen[1], ShouldBeTrue)
		})
	})

	Convey("Given a temperature near zero (very low)", t, func() {
		runner := New(7)
		declaration := program.SamplerDeclaration{
			ID:     "argmax",
			Config: map[string]any{"temperature": 0.0001, "top_k": 1},
		}

		Convey("It should always pick the argmax token", func() {
			logits := []float64{0.1, 0.2, 5.0, 0.4}

			for index := 0; index < 16; index++ {
				token, _, err := runner.Next(context.Background(), declaration, logits, nil)
				So(err, ShouldBeNil)
				So(token, ShouldEqual, 2)
			}
		})
	})

	Convey("Given a stop_token_ids list", t, func() {
		runner := New(42)
		declaration := program.SamplerDeclaration{
			ID: "stop",
			Config: map[string]any{
				"temperature":    1.0,
				"stop_token_ids": []int{2},
			},
		}

		Convey("It should set stopped when the chosen token matches", func() {
			logits := []float64{math.Inf(-1), math.Inf(-1), 1.0}
			token, stopped, err := runner.Next(context.Background(), declaration, logits, nil)
			So(err, ShouldBeNil)
			So(token, ShouldEqual, 2)
			So(stopped, ShouldBeTrue)
		})
	})

	Convey("Given a sampler with no logits", t, func() {
		runner := New(1)
		declaration := program.SamplerDeclaration{ID: "x"}

		_, _, err := runner.Next(context.Background(), declaration, nil, nil)
		So(err, ShouldNotBeNil)
	})

	Convey("Given two runners with the same baseSeed and id", t, func() {
		Convey("They should produce identical sequences", func() {
			runnerA := New(99)
			runnerB := New(99)
			declaration := program.SamplerDeclaration{ID: "seq", Config: map[string]any{"temperature": 1.0}}
			logits := []float64{0.1, 0.3, 0.6, 1.2}

			for index := 0; index < 8; index++ {
				tokenA, _, _ := runnerA.Next(context.Background(), declaration, logits, nil)
				tokenB, _, _ := runnerB.Next(context.Background(), declaration, logits, nil)
				So(tokenA, ShouldEqual, tokenB)
			}
		})
	})
}
