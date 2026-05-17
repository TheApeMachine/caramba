package diffusion

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestFlowMatchEulerScheduler_Step(test *testing.T) {
	Convey("Given a FlowMatch Euler scheduler without timestep shifting", test, func() {
		scheduler, err := NewFlowMatchEulerScheduler(SchedulerConfig{
			Type:              "flow_match_euler_discrete",
			Steps:             3,
			NumTrainTimesteps: 1000,
			Shift:             1,
			BaseImageSeqLen:   256,
			MaxImageSeqLen:    4096,
			BaseShift:         0.5,
			MaxShift:          1.15,
			TimeShiftType:     "exponential",
		}, 256)
		So(err, ShouldBeNil)

		Convey("It should expose sigma-scaled timesteps", func() {
			timesteps := scheduler.Timesteps()

			So(timesteps[0], ShouldAlmostEqual, 1000)
			So(timesteps[1], ShouldAlmostEqual, 666.6666666666667)
			So(timesteps[2], ShouldAlmostEqual, 333.33333333333337)
		})

		Convey("It should apply the Euler delta between adjacent sigmas", func() {
			next, err := scheduler.Step(0, []float64{0, 1}, []float64{2, 4})

			So(err, ShouldBeNil)
			So(next[0], ShouldAlmostEqual, -0.6666666666666666)
			So(next[1], ShouldAlmostEqual, -0.33333333333333326)
		})
	})

	Convey("Given an unsupported dynamic shift type", test, func() {
		_, err := NewFlowMatchEulerScheduler(SchedulerConfig{
			Type:              "flow_match_euler_discrete",
			Steps:             1,
			NumTrainTimesteps: 1000,
			TimeShiftType:     "unsupported",
		}, 1)

		Convey("It should reject the scheduler configuration", func() {
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "unsupported scheduler time_shift_type")
		})
	})
}

func BenchmarkFlowMatchEulerScheduler_Step(benchmark *testing.B) {
	scheduler, err := NewFlowMatchEulerScheduler(SchedulerConfig{
		Type:              "flow_match_euler_discrete",
		Steps:             4,
		NumTrainTimesteps: 1000,
		Shift:             1,
		BaseImageSeqLen:   256,
		MaxImageSeqLen:    4096,
		BaseShift:         0.5,
		MaxShift:          1.15,
		TimeShiftType:     "exponential",
	}, 256)

	if err != nil {
		benchmark.Fatal(err)
	}

	latents := []float64{0, 1, 2, 3}
	output := []float64{1, 1, 1, 1}

	for benchmark.Loop() {
		if _, err := scheduler.Step(0, latents, output); err != nil {
			benchmark.Fatal(err)
		}
	}
}
