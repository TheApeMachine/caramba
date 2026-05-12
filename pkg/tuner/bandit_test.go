package tuner

import (
	"math/rand"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestBandit(t *testing.T) {
	Convey("Given a new bandit with three arms", t, func() {
		arms := banditTestArms()
		bandit, err := NewBandit("active_inference", arms, 13)

		So(err, ShouldBeNil)
		So(bandit, ShouldNotBeNil)

		Convey("When SelectArm is called three times without Update", func() {
			selected := map[string]bool{}

			for range arms {
				arm, _, err := bandit.SelectArm()

				So(err, ShouldBeNil)
				So(selected[arm.ID], ShouldBeFalse)

				selected[arm.ID] = true
				So(bandit.Update(arm.ID, 10.0), ShouldBeNil)
			}

			Convey("It returns each arm exactly once", func() {
				So(selected, ShouldHaveLength, len(arms))
				So(selected["arm_1"], ShouldBeTrue)
				So(selected["arm_2"], ShouldBeTrue)
				So(selected["arm_3"], ShouldBeTrue)
			})
		})
	})

	Convey("Given three arms have all been pulled once", t, func() {
		bandit, err := NewBandit("active_inference", banditTestArms(), 1)

		So(err, ShouldBeNil)
		So(bandit.Update("arm_1", 10), ShouldBeNil)
		So(bandit.Update("arm_2", 5), ShouldBeNil)
		So(bandit.Update("arm_3", 8), ShouldBeNil)

		Convey("When SelectArm is called repeatedly", func() {
			selectedArmTwo := 0

			for range 1000 {
				arm, _, err := bandit.SelectArm()

				So(err, ShouldBeNil)

				if arm.ID == "arm_2" {
					selectedArmTwo++
				}
			}

			Convey("It samples arm_2 more than half the time", func() {
				So(selectedArmTwo, ShouldBeGreaterThan, 500)
			})
		})
	})

	Convey("Given Update is called with an unknown arm ID", t, func() {
		bandit, err := NewBandit("active_inference", banditTestArms(), 1)

		So(err, ShouldBeNil)

		err = bandit.Update("missing", 1.0)

		Convey("It returns an error", func() {
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "unknown arm ID")
		})
	})

	Convey("Given metrics for each arm are drawn from gaussians with known means", t, func() {
		bandit, err := NewBandit("active_inference", banditTestArms(), 21)
		random := rand.New(rand.NewSource(34))

		So(err, ShouldBeNil)

		for range 200 {
			arm, _, err := bandit.SelectArm()

			So(err, ShouldBeNil)
			So(bandit.Update(arm.ID, gaussianMetric(random, arm.ID)), ShouldBeNil)
		}

		Convey("After 200 trials the posterior assigns most mass to the lowest true mean", func() {
			So(bandit.posterior[1], ShouldBeGreaterThan, 0.5)
		})
	})

	Convey("Given NewBandit is called with strategy ucb1", t, func() {
		_, err := NewBandit("ucb1", banditTestArms(), 1)

		Convey("It returns an error", func() {
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "unsupported bandit strategy")
		})
	})

	Convey("Given NewBandit is called with no arms", t, func() {
		_, err := NewBandit("active_inference", []Arm{}, 1)

		Convey("It returns an error", func() {
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "0 arms")
		})
	})
}

func BenchmarkSelectArm(b *testing.B) {
	bandit, err := NewBandit("active_inference", banditTestArms(), 1)

	if err != nil {
		b.Fatal(err)
	}

	for _, observation := range []struct {
		armID  string
		metric float64
	}{
		{armID: "arm_1", metric: 10},
		{armID: "arm_2", metric: 5},
		{armID: "arm_3", metric: 8},
	} {
		if err := bandit.Update(observation.armID, observation.metric); err != nil {
			b.Fatal(err)
		}
	}

	for b.Loop() {
		_, _, _ = bandit.SelectArm()
	}
}

func BenchmarkUpdate(b *testing.B) {
	bandit, err := NewBandit("active_inference", banditTestArms(), 1)

	if err != nil {
		b.Fatal(err)
	}

	metrics := []float64{10, 5, 8}
	arms := banditTestArms()
	index := 0

	for b.Loop() {
		arm := arms[index%len(arms)]

		if err := bandit.Update(arm.ID, metrics[index%len(metrics)]); err != nil {
			b.Fatal(err)
		}

		index++
	}
}

func banditTestArms() []Arm {
	return []Arm{
		{ID: "arm_1", Config: map[string]any{"lr": 1e-4}},
		{ID: "arm_2", Config: map[string]any{"lr": 3e-4}},
		{ID: "arm_3", Config: map[string]any{"lr": 1e-3}},
	}
}

func gaussianMetric(random *rand.Rand, armID string) float64 {
	means := map[string]float64{
		"arm_1": 10.0,
		"arm_2": 4.0,
		"arm_3": 8.0,
	}

	return means[armID] + random.NormFloat64()*0.25
}
