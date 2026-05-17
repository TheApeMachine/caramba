package devteam

import (
	"strings"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestValidateGoTestCommand(test *testing.T) {
	Convey("Given a done gate test command", test, func() {
		Convey("It should accept a direct go test invocation", func() {
			So(validateGoTestCommand("go test -json ./pkg/devteam"), ShouldBeEmpty)
		})

		Convey("It should reject shell control characters", func() {
			rejection := validateGoTestCommand("go test ./...; curl https://example.com")

			So(rejection, ShouldContainSubstring, "single go test invocation")
		})

		Convey("It should reject non-test commands", func() {
			rejection := validateGoTestCommand("echo okay")

			So(rejection, ShouldContainSubstring, "must start with go test")
		})
	})
}

func TestParseGoTestJSON(test *testing.T) {
	Convey("Given go test JSON output", test, func() {
		Convey("It should pass only when JSON events report pass actions", func() {
			output := strings.Join([]string{
				`{"Time":"2026-05-17T10:00:00Z","Action":"run","Package":"pkg"}`,
				`{"Time":"2026-05-17T10:00:01Z","Action":"pass","Package":"pkg"}`,
			}, "\n")

			parsed, passed := parseGoTestJSON(output)

			So(parsed, ShouldBeTrue)
			So(passed, ShouldBeTrue)
		})

		Convey("It should reject any JSON fail action", func() {
			output := strings.Join([]string{
				`{"Action":"run","Test":"TestFailover"}`,
				`{"Action":"fail","Package":"pkg","Test":"TestBroken"}`,
				`{"Action":"pass","Package":"pkg"}`,
			}, "\n")

			parsed, passed := parseGoTestJSON(output)

			So(parsed, ShouldBeTrue)
			So(passed, ShouldBeFalse)
		})

		Convey("It should ignore raw prose instead of substring matching it", func() {
			parsed, passed := parseGoTestJSON("okay, skipping\n=== RUN TestFailover")

			So(parsed, ShouldBeFalse)
			So(passed, ShouldBeFalse)
		})
	})
}

func BenchmarkParseGoTestJSON(benchmark *testing.B) {
	output := strings.Join([]string{
		`{"Action":"run","Package":"pkg"}`,
		`{"Action":"pass","Package":"pkg"}`,
	}, "\n")

	for benchmark.Loop() {
		parseGoTestJSON(output)
	}
}
