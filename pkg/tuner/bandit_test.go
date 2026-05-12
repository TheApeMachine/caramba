package tuner

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestBandit(t *testing.T) {
	Convey("Given a Bandit tuner with active_inference strategy", t, func() {
		arms := []Arm{
			{ID: "arm_1", Config: map[string]any{"lr": 1e-4}},
			{ID: "arm_2", Config: map[string]any{"lr": 3e-4}},
			{ID: "arm_3", Config: map[string]any{"lr": 1e-3}},
		}

		bandit, err := NewBandit("active_inference", arms)
		So(err, ShouldBeNil)
		So(bandit, ShouldNotBeNil)

		Convey("When attempting to create a bandit with no arms", func() {
			_, err := NewBandit("active_inference", []Arm{})

			Convey("It should return an error", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "0 arms")
			})
		})

		Convey("When selecting an arm with valid state probabilities", func() {
			// 2 states (e.g., success, failure), 3 arms
			// state 0 probabilities for arms 1, 2, 3
			// state 1 probabilities for arms 1, 2, 3
			stateProbs := []float64{
				0.1, 0.5, 0.9,
				0.9, 0.5, 0.1,
			}

			selectedArm, score, err := bandit.SelectArm(stateProbs, 2)

			Convey("It should succeed and return the arm with the minimum expected free energy", func() {
				So(err, ShouldBeNil)
				So(selectedArm.ID, ShouldNotBeEmpty)
				So(score, ShouldNotEqual, 0.0)
			})
		})

		Convey("When providing invalid state probabilities", func() {
			stateProbs := []float64{0.5, 0.5} // length 2 != 2*3=6

			_, _, err := bandit.SelectArm(stateProbs, 2)

			Convey("It should return an error", func() {
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "must equal numStates*numArms")
			})
		})
	})

	Convey("Given a Bandit tuner with an unknown strategy", t, func() {
		arms := []Arm{
			{ID: "fallback_arm", Config: map[string]any{"batch_size": 32}},
		}

		bandit, err := NewBandit("random", arms)
		So(err, ShouldBeNil)

		Convey("When selecting an arm", func() {
			stateProbs := []float64{0.5, 0.5} // length 2 for 2 states, 1 arm
			selectedArm, score, err := bandit.SelectArm(stateProbs, 2)

			Convey("It should fallback to the first arm with score 0.0", func() {
				So(err, ShouldBeNil)
				So(selectedArm.ID, ShouldEqual, "fallback_arm")
				So(score, ShouldEqual, 0.0)
			})
		})
	})
}
