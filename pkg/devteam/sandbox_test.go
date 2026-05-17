package devteam

import (
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
)

func TestSandboxRedact(test *testing.T) {
	Convey("Given a sandbox with a GitHub token", test, func() {
		sandbox := &Sandbox{cfg: SandboxConfig{GitHubToken: "ghp_secret/token"}}

		Convey("It should remove raw and escaped secrets from text", func() {
			text := `git push https://ghp_secret/token@github.com/owner/repo ghp_secret%2Ftoken`
			redacted := sandbox.Redact(text)

			So(redacted, ShouldNotContainSubstring, "ghp_secret/token")
			So(redacted, ShouldNotContainSubstring, "ghp_secret%2Ftoken")
			So(redacted, ShouldContainSubstring, "[REDACTED]")
		})
	})
}

func TestCombinedOutput(test *testing.T) {
	Convey("Given stdout and stderr content", test, func() {
		Convey("It should preserve both streams without Docker frame headers", func() {
			output := combinedOutput("stdout", "stderr")

			So(output, ShouldEqual, "stdout\nstderr")
		})

		Convey("It should return one stream without an extra separator", func() {
			So(combinedOutput("stdout", ""), ShouldEqual, "stdout")
			So(combinedOutput("", "stderr"), ShouldEqual, "stderr")
		})
	})
}

func TestSandboxDefaults(test *testing.T) {
	Convey("Given a sandbox without explicit resource settings", test, func() {
		sandbox := &Sandbox{}

		Convey("It should apply bounded defaults", func() {
			So(sandbox.execTimeout(), ShouldEqual, sandboxDefaultExecTimeout)
			So(sandbox.memoryBytes(), ShouldEqual, int64(sandboxDefaultMemoryBytes))
			So(sandbox.nanoCPUs(), ShouldEqual, int64(sandboxDefaultNanoCPUs))
			So(sandbox.pidsLimit(), ShouldEqual, int64(sandboxDefaultPidsLimit))
		})

		Convey("It should build a restricted Docker host config", func() {
			hostConfig := sandbox.hostConfig()

			So(hostConfig.ReadonlyRootfs, ShouldBeTrue)
			So(string(hostConfig.NetworkMode), ShouldEqual, "none")
			So(hostConfig.Memory, ShouldEqual, int64(sandboxDefaultMemoryBytes))
			So(hostConfig.NanoCPUs, ShouldEqual, int64(sandboxDefaultNanoCPUs))
			So(*hostConfig.PidsLimit, ShouldEqual, int64(sandboxDefaultPidsLimit))
			So(hostConfig.CapDrop, ShouldContain, "ALL")
			So(hostConfig.Tmpfs, ShouldContainKey, "/workspace")
		})
	})

	Convey("Given a sandbox with explicit resource settings", test, func() {
		sandbox := &Sandbox{cfg: SandboxConfig{
			ExecTimeout: 30 * time.Second,
			MemoryBytes: 512 << 20,
			NanoCPUs:    500_000_000,
			PidsLimit:   64,
		}}

		Convey("It should use the configured bounds", func() {
			So(sandbox.execTimeout(), ShouldEqual, 30*time.Second)
			So(sandbox.memoryBytes(), ShouldEqual, int64(512<<20))
			So(sandbox.nanoCPUs(), ShouldEqual, int64(500_000_000))
			So(sandbox.pidsLimit(), ShouldEqual, int64(64))
		})
	})
}

func BenchmarkSandboxRedact(benchmark *testing.B) {
	sandbox := &Sandbox{cfg: SandboxConfig{GitHubToken: "ghp_secret"}}

	for benchmark.Loop() {
		_ = sandbox.Redact("git error ghp_secret")
	}
}
