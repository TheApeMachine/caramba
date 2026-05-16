package cmd

import (
	"testing"

	bubbleprogress "charm.land/bubbles/v2/progress"
	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/qpool"
)

func TestQPoolProgressModel_Update(test *testing.T) {
	Convey("Given qpool progress model Update", test, func() {
		progressModel := qpoolProgressModel{
			progress: bubbleprogress.New(bubbleprogress.WithDefaultBlend()),
		}
		event := qpool.NewInfoEvent(
			"diffusion",
			"denoise.step",
			"running denoiser step",
			[]qpool.Field{
				{Key: "step", Value: 2},
				{Key: "steps", Value: 4},
			},
		)

		Convey("It should update visible status from qpool events", func() {
			updatedModel, command := progressModel.Update(qpoolEventMsg{event: event})
			typedModel := updatedModel.(qpoolProgressModel)

			So(command, ShouldNotBeNil)
			So(typedModel.active, ShouldBeTrue)
			So(typedModel.status, ShouldEqual, "running denoiser step")
			So(typedModel.progress.Percent(), ShouldAlmostEqual, 0.5, 1e-9)
		})
	})
}

func TestQPoolProgressFromEvent(test *testing.T) {
	Convey("Given qpoolProgressFromEvent", test, func() {
		Convey("It should derive byte progress from Hub events", func() {
			event := qpool.NewInfoEvent(
				"hub",
				"download.http.read",
				"downloading",
				[]qpool.Field{
					{Key: "file", Value: "model.safetensors"},
					{Key: "read_bytes", Value: int64(64)},
					{Key: "expected_bytes", Value: int64(128)},
				},
			)

			state := qpoolProgressFromEvent(event, 0)

			So(state.known, ShouldBeTrue)
			So(state.percent, ShouldAlmostEqual, 0.5, 1e-9)
			So(state.detail, ShouldEqual, "model.safetensors")
		})

		Convey("It should complete progress from done events", func() {
			event := qpool.NewInfoEvent(
				"weights",
				"open.ready",
				"SafeTensors store ready",
				[]qpool.Field{{Key: "done", Value: true}},
			)

			state := qpoolProgressFromEvent(event, 0.42)

			So(state.known, ShouldBeTrue)
			So(state.percent, ShouldAlmostEqual, 1, 1e-9)
		})

		Convey("It should advance unknown progress without completing it", func() {
			event := qpool.NewInfoEvent("chat", "manifest.resolve", "resolving", nil)

			state := qpoolProgressFromEvent(event, 0.9)

			So(state.known, ShouldBeFalse)
			So(state.percent, ShouldAlmostEqual, 0.93, 1e-9)
		})
	})
}

func BenchmarkQPoolProgressFromEvent(benchmark *testing.B) {
	event := qpool.NewInfoEvent(
		"chat",
		"generation.token",
		"generating",
		[]qpool.Field{
			{Key: "token", Value: 12},
			{Key: "tokens", Value: 128},
		},
	)

	for benchmark.Loop() {
		_ = qpoolProgressFromEvent(event, 0)
	}
}
