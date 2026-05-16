//go:build cgo && xla

package xla

import (
	"fmt"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	cpuvsa "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/vsa"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestXLAVSAOps_Bundle(test *testing.T) {
	platform := xlaPJRTAvailablePlatform(test)

	Convey("Given an XLA VSA runtime", test, func() {
		xlaVSAOps, err := NewVSAOps(platform)
		So(err, ShouldBeNil)
		defer xlaVSAOps.Shutdown()

		cpuBundle := cpuvsa.NewBundle()

		Convey("It should match CPU bundle execution at contract sizes", func() {
			for _, length := range []int{1, 7, 64, 1024, 8192} {
				first := xlaPattern(length, 0.17, 0.013)
				second := xlaPattern(length, -0.09, 0.019)
				third := xlaPattern(length, 0.04, 0.007)
				cpuState := state.NewDict().
					WithShape([]int{length}).
					WithInputs(first, second, third)

				cpuUpdated, err := cpuBundle.Forward(cpuState)
				So(err, ShouldBeNil)

				xlaOut, err := xlaVSAOps.Bundle([]int{length}, first, second, third)
				So(err, ShouldBeNil)

				assertXLASlice(
					fmt.Sprintf("vsa.bundle/%d", length),
					xlaOut,
					cpuUpdated.Out,
					1e-9,
				)
			}
		})
	})
}

func TestXLAVSAOps_Similarity(test *testing.T) {
	platform := xlaPJRTAvailablePlatform(test)

	Convey("Given an XLA VSA runtime", test, func() {
		xlaVSAOps, err := NewVSAOps(platform)
		So(err, ShouldBeNil)
		defer xlaVSAOps.Shutdown()

		cpuSimilarity := cpuvsa.NewSimilarity()

		Convey("It should match CPU similarity execution at contract sizes", func() {
			for _, length := range []int{1, 7, 64, 1024, 8192} {
				left := xlaPattern(length, 0.29, 0.011)
				right := xlaPattern(length, -0.13, 0.017)
				cpuState := state.NewDict().
					WithShape([]int{length}).
					WithInputs(left, right)

				cpuUpdated, err := cpuSimilarity.Forward(cpuState)
				So(err, ShouldBeNil)

				xlaOut, err := xlaVSAOps.Similarity([]int{length}, left, right)
				So(err, ShouldBeNil)

				assertXLASlice(
					fmt.Sprintf("vsa.similarity/%d", length),
					xlaOut,
					cpuUpdated.Out,
					1e-9,
				)
			}
		})
	})
}

func BenchmarkXLAVSAOps_Bundle(benchmark *testing.B) {
	platform := xlaPJRTAvailablePlatform(benchmark)
	xlaVSAOps, err := NewVSAOps(platform)
	if err != nil {
		benchmark.Fatal(err)
	}
	defer xlaVSAOps.Shutdown()

	first := xlaPattern(8192, 0.17, 0.013)
	second := xlaPattern(8192, -0.09, 0.019)
	third := xlaPattern(8192, 0.04, 0.007)

	benchmark.ResetTimer()

	for benchmark.Loop() {
		if _, err := xlaVSAOps.Bundle([]int{8192}, first, second, third); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkXLAVSAOps_Similarity(benchmark *testing.B) {
	platform := xlaPJRTAvailablePlatform(benchmark)
	xlaVSAOps, err := NewVSAOps(platform)
	if err != nil {
		benchmark.Fatal(err)
	}
	defer xlaVSAOps.Shutdown()

	left := xlaPattern(8192, 0.29, 0.011)
	right := xlaPattern(8192, -0.13, 0.017)

	benchmark.ResetTimer()

	for benchmark.Loop() {
		if _, err := xlaVSAOps.Similarity([]int{8192}, left, right); err != nil {
			benchmark.Fatal(err)
		}
	}
}
