package rotary

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestConfig_InverseFrequencies(test *testing.T) {
	Convey("Given a rotary embedding configuration", test, func() {
		Convey("It should compute default inverse frequencies", func() {
			frequencies, err := (Config{Base: 10000}).InverseFrequencies(8)

			So(err, ShouldBeNil)
			So(frequencies, ShouldHaveLength, 4)
			So(frequencies[0], ShouldAlmostEqual, 1.0)
			So(frequencies[1], ShouldAlmostEqual, 0.1)
			So(frequencies[2], ShouldAlmostEqual, 0.01)
			So(frequencies[3], ShouldAlmostEqual, 0.001)
		})

		Convey("It should apply Llama 3 low and medium frequency scaling", func() {
			config := Config{
				Base:                          500000,
				Type:                          TypeLlama3,
				Factor:                        32,
				LowFreqFactor:                 1,
				HighFreqFactor:                4,
				OriginalMaxPositionEmbeddings: 8192,
			}

			frequencies, err := config.InverseFrequencies(64)

			So(err, ShouldBeNil)
			So(frequencies[0], ShouldAlmostEqual, 1.0)

			defaultMid := defaultInverseFrequency(500000, 16, 64)
			So(frequencies[16], ShouldBeLessThan, defaultMid)
			So(frequencies[16], ShouldBeGreaterThan, defaultMid/32)

			defaultLow := defaultInverseFrequency(500000, 31, 64)
			So(frequencies[31], ShouldAlmostEqual, defaultLow/32)
		})

		Convey("It should reject unsupported RoPE types", func() {
			_, err := (Config{Type: "linear"}).InverseFrequencies(64)

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "unsupported")
		})
	})
}

func BenchmarkConfig_InverseFrequencies(benchmark *testing.B) {
	config := Config{
		Base:                          500000,
		Type:                          TypeLlama3,
		Factor:                        32,
		LowFreqFactor:                 1,
		HighFreqFactor:                4,
		OriginalMaxPositionEmbeddings: 8192,
	}

	for benchmark.Loop() {
		frequencies, err := config.InverseFrequencies(64)

		if err != nil || math.IsNaN(frequencies[0]) {
			benchmark.Fatal(err)
		}
	}
}
