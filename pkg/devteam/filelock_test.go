package devteam

import (
	"context"
	"fmt"
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/qpool"
)

func TestFileLockRegistryClaim(t *testing.T) {
	Convey("Given a running FileLockRegistry", t, func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		registry := NewFileLockRegistry(ctx)
		defer registry.Close()

		Convey("It should grant a claim to the first agent that requests a path", func() {
			result := registry.Claim("agent-a", "pkg/foo/bar.go", "add error handling")

			So(result.Acquired, ShouldBeTrue)
			So(result.HolderID, ShouldBeEmpty)
		})

		Convey("It should deny a claim when another agent already holds the path", func() {
			registry.Claim("agent-a", "pkg/foo/bar.go", "add error handling")
			result := registry.Claim("agent-b", "pkg/foo/bar.go", "refactor loop")

			So(result.Acquired, ShouldBeFalse)
			So(result.HolderID, ShouldEqual, "agent-a")
			So(result.Intent, ShouldEqual, "add error handling")
		})

		Convey("It should allow the same agent to re-claim a path it already holds", func() {
			registry.Claim("agent-a", "pkg/foo/bar.go", "add error handling")
			result := registry.Claim("agent-a", "pkg/foo/bar.go", "updated intent")

			So(result.Acquired, ShouldBeTrue)
		})

		Convey("It should grant a claim after the original holder releases it", func() {
			registry.Claim("agent-a", "pkg/foo/bar.go", "add error handling")
			registry.Release("agent-a", "pkg/foo/bar.go")
			result := registry.Claim("agent-b", "pkg/foo/bar.go", "refactor loop")

			So(result.Acquired, ShouldBeTrue)
		})
	})
}

func TestFileLockRegistryRelease(t *testing.T) {
	Convey("Given a registry with claims held by two different agents", t, func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		registry := NewFileLockRegistry(ctx)
		defer registry.Close()

		registry.Claim("agent-a", "pkg/foo/a.go", "intent a1")
		registry.Claim("agent-a", "pkg/foo/b.go", "intent a2")
		registry.Claim("agent-b", "pkg/bar/c.go", "intent b1")

		Convey("It should release only the specific path when Release is called", func() {
			registry.Release("agent-a", "pkg/foo/a.go")

			result := registry.Claim("agent-b", "pkg/foo/a.go", "take over")
			So(result.Acquired, ShouldBeTrue)

			// agent-a still holds b.go
			result2 := registry.Claim("agent-b", "pkg/foo/b.go", "take over b")
			So(result2.Acquired, ShouldBeFalse)
			So(result2.HolderID, ShouldEqual, "agent-a")
		})

		Convey("It should release all paths held by an agent when ReleaseAll is called", func() {
			registry.ReleaseAll("agent-a")

			resultA := registry.Claim("agent-b", "pkg/foo/a.go", "take a")
			resultB := registry.Claim("agent-b", "pkg/foo/b.go", "take b")

			So(resultA.Acquired, ShouldBeTrue)
			So(resultB.Acquired, ShouldBeTrue)

			// agent-b's own claim should be unaffected
			snap := registry.Snapshot()
			So(snap, ShouldContainKey, "pkg/bar/c.go")
		})

		Convey("It should not release a path that belongs to a different agent", func() {
			registry.Release("agent-b", "pkg/foo/a.go") // agent-b does not own this

			result := registry.Claim("agent-b", "pkg/foo/a.go", "steal")
			So(result.Acquired, ShouldBeFalse)
			So(result.HolderID, ShouldEqual, "agent-a")
		})
	})
}

func TestFileLockRegistrySnapshot(t *testing.T) {
	Convey("Given a registry with several active claims", t, func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		registry := NewFileLockRegistry(ctx)
		defer registry.Close()

		registry.Claim("agent-a", "pkg/foo/a.go", "intent-a")
		registry.Claim("agent-b", "pkg/bar/b.go", "intent-b")

		Convey("It should return a snapshot containing all claimed paths", func() {
			snap := registry.Snapshot()

			So(snap, ShouldContainKey, "pkg/foo/a.go")
			So(snap, ShouldContainKey, "pkg/bar/b.go")
			So(snap["pkg/foo/a.go"], ShouldContainSubstring, "agent-a")
			So(snap["pkg/foo/a.go"], ShouldContainSubstring, "intent-a")
		})

		Convey("It should reflect releases in the snapshot", func() {
			registry.Release("agent-a", "pkg/foo/a.go")
			snap := registry.Snapshot()

			So(snap, ShouldNotContainKey, "pkg/foo/a.go")
			So(snap, ShouldContainKey, "pkg/bar/b.go")
		})
	})
}

func TestFileLockRegistryConcurrency(t *testing.T) {
	Convey("Given 20 qpool jobs competing to claim the same path", t, func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		registry := NewFileLockRegistry(ctx)
		defer registry.Close()

		const workers = 20
		results := make([]ClaimResult, workers)
		pool := qpool.NewQ(ctx, workers, workers, &qpool.Config{
			SchedulingTimeout:  time.Second,
			JobChannelCapacity: workers,
			Scaler:             nil,
		})
		defer pool.Close()

		resultChannels := make([]chan *qpool.QValue, workers)

		for index := range workers {
			index := index

			resultChannels[index] = pool.Schedule(
				fmt.Sprintf("filelock-claim-%d", index),
				func(context.Context) (any, error) {
					agentID := fmt.Sprintf("agent-%d", index)

					return registry.Claim(agentID, "shared/file.go", "concurrent claim"), nil
				},
				qpool.WithExecTimeout(time.Second),
			)
		}

		for index, resultChannel := range resultChannels {
			result := <-resultChannel
			So(result.Error, ShouldBeNil)
			results[index] = result.Value.(ClaimResult)
		}

		Convey("It should grant the claim to exactly one agent", func() {
			acquired := 0

			for _, result := range results {
				if result.Acquired {
					acquired++
				}
			}

			So(acquired, ShouldEqual, 1)
		})
	})
}

func TestFileLockRegistryClose(t *testing.T) {
	Convey("Given a closed FileLockRegistry", t, func() {
		ctx := context.Background()
		registry := NewFileLockRegistry(ctx)
		registry.Close()

		Convey("It should return Acquired=false for any new claim", func() {
			result := registry.Claim("agent-a", "any/path.go", "intent")

			So(result.Acquired, ShouldBeFalse)
		})

		Convey("It should be safe to call Close again", func() {
			So(func() { registry.Close() }, ShouldNotPanic)
		})
	})
}

func BenchmarkFileLockRegistryClaim(b *testing.B) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	registry := NewFileLockRegistry(ctx)
	defer registry.Close()

	for b.Loop() {
		registry.Claim("agent-bench", "pkg/foo/bar.go", "bench intent")
		registry.Release("agent-bench", "pkg/foo/bar.go")
	}
}

func BenchmarkFileLockRegistryReleaseAll(b *testing.B) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	registry := NewFileLockRegistry(ctx)
	defer registry.Close()

	paths := []string{
		"pkg/a/a.go", "pkg/b/b.go", "pkg/c/c.go",
		"pkg/d/d.go", "pkg/e/e.go",
	}

	for b.Loop() {
		for _, path := range paths {
			registry.Claim("agent-bench", path, "bench")
		}

		registry.ReleaseAll("agent-bench")
	}
}
