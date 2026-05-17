package state

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestRegistry(t *testing.T) {
	Convey("Given a fresh state registry with the built-in factories", t, func() {
		registry := NewRegistry()
		So(registry.Register("token_buffer", newTokenBufferFromConfig), ShouldBeNil)
		So(registry.Register("counter", newCounterFromConfig), ShouldBeNil)

		Convey("It should build a token_buffer with the supplied id", func() {
			built, err := registry.Build("token_buffer", "history", nil)
			So(err, ShouldBeNil)
			So(built.ID(), ShouldEqual, "history")
			So(built.Type(), ShouldEqual, "token_buffer")
		})

		Convey("It should reject duplicate registrations", func() {
			err := registry.Register("counter", newCounterFromConfig)
			So(err, ShouldNotBeNil)
		})

		Convey("It should reject unknown types", func() {
			_, err := registry.Build("missing", "x", nil)
			So(err, ShouldNotBeNil)
		})

		Convey("It should reject empty ids", func() {
			_, err := registry.Build("counter", "", nil)
			So(err, ShouldNotBeNil)
		})

		Convey("Types should return a sorted list", func() {
			types := registry.Types()
			So(types, ShouldResemble, []string{"counter", "token_buffer"})
		})

		Convey("Default registry should include token_buffer, counter, rng, tensor, kv_cache", func() {
			defaults := Default.Types()
			So(defaults, ShouldContain, "token_buffer")
			So(defaults, ShouldContain, "counter")
			So(defaults, ShouldContain, "rng")
			So(defaults, ShouldContain, "tensor")
			So(defaults, ShouldContain, "kv_cache")
		})
	})
}

func TestTokenBuffer(t *testing.T) {
	Convey("Given a fresh TokenBuffer", t, func() {
		ctx := context.Background()
		buffer := newTokenBuffer("history")

		Convey("Append and Extend should grow the buffer", func() {
			buffer.Append(1)
			buffer.Extend([]int{2, 3, 4})

			So(buffer.Length(), ShouldEqual, 4)
			So(buffer.Tokens(), ShouldResemble, []int{1, 2, 3, 4})
		})

		Convey("Reset should clear the buffer", func() {
			buffer.Extend([]int{1, 2, 3})
			So(buffer.Reset(ctx), ShouldBeNil)
			So(buffer.Length(), ShouldEqual, 0)
		})

		Convey("Snapshot then Restore on a fresh buffer should match", func() {
			buffer.Extend([]int{5, 6, 7, 8})
			snapshot, err := buffer.Snapshot(ctx)
			So(err, ShouldBeNil)

			restored := newTokenBuffer("history")
			So(restored.Restore(ctx, snapshot), ShouldBeNil)
			So(restored.Tokens(), ShouldResemble, []int{5, 6, 7, 8})
		})

		Convey("Inspect should report length", func() {
			buffer.Extend([]int{1, 2})
			inspection, err := buffer.Inspect(ctx)
			So(err, ShouldBeNil)
			So(inspection.Values["length"], ShouldEqual, 2)
		})
	})
}

func TestCounter(t *testing.T) {
	Convey("Given a Counter initialized at 0", t, func() {
		ctx := context.Background()
		counter := newCounter("position", 0)

		Convey("Increment should advance and return the new value", func() {
			So(counter.Increment(1), ShouldEqual, 1)
			So(counter.Increment(3), ShouldEqual, 4)
			So(counter.Value(), ShouldEqual, 4)
		})

		Convey("Set should replace and Reset should go back to initial", func() {
			counter.Set(10)
			So(counter.Value(), ShouldEqual, 10)
			So(counter.Reset(ctx), ShouldBeNil)
			So(counter.Value(), ShouldEqual, 0)
		})

		Convey("Snapshot then Restore should round-trip the value", func() {
			counter.Set(42)
			snapshot, err := counter.Snapshot(ctx)
			So(err, ShouldBeNil)

			restored := newCounter("position", 0)
			So(restored.Restore(ctx, snapshot), ShouldBeNil)
			So(restored.Value(), ShouldEqual, 42)
		})
	})
}

func TestRNG(t *testing.T) {
	Convey("Given two RNGs seeded identically", t, func() {
		ctx := context.Background()
		rngA := newRNG("rng-a", 12345)
		rngB := newRNG("rng-b", 12345)

		Convey("They should produce identical sequences", func() {
			for index := 0; index < 16; index++ {
				So(rngA.Float64(), ShouldEqual, rngB.Float64())
			}
		})

		Convey("Restore should reproduce the same state mid-stream", func() {
			for index := 0; index < 7; index++ {
				_ = rngA.Float64()
			}

			snapshot, err := rngA.Snapshot(ctx)
			So(err, ShouldBeNil)

			restored := newRNG("rng-c", 0)
			So(restored.Restore(ctx, snapshot), ShouldBeNil)

			expectedNext := rngA.Float64()
			restoredNext := restored.Float64()
			So(restoredNext, ShouldEqual, expectedNext)
		})
	})
}

func TestTensor(t *testing.T) {
	Convey("Given a Tensor of shape [2,3]", t, func() {
		ctx := context.Background()
		tensor := newTensor("latents", []int{2, 3})

		Convey("It should allocate 6 zeros", func() {
			So(tensor.Values(), ShouldResemble, []float64{0, 0, 0, 0, 0, 0})
		})

		Convey("Set should replace contents when the shape product matches", func() {
			err := tensor.Set([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
			So(err, ShouldBeNil)
			So(tensor.Values(), ShouldResemble, []float64{1, 2, 3, 4, 5, 6})
		})

		Convey("Set should reject mismatched lengths", func() {
			err := tensor.Set([]int{2, 3}, []float64{1, 2, 3})
			So(err, ShouldNotBeNil)
		})

		Convey("Snapshot then Restore should round-trip shape and contents", func() {
			So(tensor.Set([]int{2, 2}, []float64{1.5, -2.5, 3.25, 4.0625}), ShouldBeNil)
			snapshot, err := tensor.Snapshot(ctx)
			So(err, ShouldBeNil)

			restored := newTensor("clone", []int{1})
			So(restored.Restore(ctx, snapshot), ShouldBeNil)
			So(restored.Shape(), ShouldResemble, []int{2, 2})
			So(restored.Values(), ShouldResemble, []float64{1.5, -2.5, 3.25, 4.0625})
		})
	})
}

func TestKVCache(t *testing.T) {
	Convey("Given a KVCache runtime state", t, func() {
		ctx := context.Background()
		built, err := Default.Build("kv_cache", "kv", map[string]any{"capacity": 128})
		So(err, ShouldBeNil)

		cacheState := built.(*KVCache)

		Convey("It should expose the underlying decoder cache", func() {
			So(cacheState.Cache(), ShouldNotBeNil)
			So(cacheState.Cache().Capacity(), ShouldEqual, 128)
		})

		Convey("Reset should advance the cache epoch and clear entries", func() {
			_, _, _, err := cacheState.Cache().Append(
				"attention",
				[]int{1, 1, 1, 1},
				[]float64{1},
				[]float64{2},
			)
			So(err, ShouldBeNil)
			before := cacheState.Cache().Epoch()

			So(cacheState.Reset(ctx), ShouldBeNil)
			So(cacheState.Cache().Epoch(), ShouldEqual, before+1)
			So(cacheState.Cache().EntryCount(), ShouldEqual, 0)
		})

		Convey("Snapshot then Restore should round-trip cache contents", func() {
			_, _, _, err := cacheState.Cache().Append(
				"attention",
				[]int{1, 1, 2, 1},
				[]float64{3, 4},
				[]float64{5, 6},
			)
			So(err, ShouldBeNil)

			snapshot, err := cacheState.Snapshot(ctx)
			So(err, ShouldBeNil)

			restoredState, err := Default.Build("kv_cache", "clone", nil)
			So(err, ShouldBeNil)

			restored := restoredState.(*KVCache)
			So(restored.Restore(ctx, snapshot), ShouldBeNil)

			inspection, err := restored.Inspect(ctx)
			So(err, ShouldBeNil)
			So(inspection.Type, ShouldEqual, "kv_cache")
			So(inspection.Values["capacity"], ShouldEqual, 128)
			So(inspection.Values["entries"], ShouldEqual, 1)
		})
	})
}

func BenchmarkKVCache_Snapshot(benchmark *testing.B) {
	built, err := Default.Build("kv_cache", "kv", map[string]any{"capacity": 128})

	if err != nil {
		benchmark.Fatal(err)
	}

	cacheState := built.(*KVCache)
	_, _, _, err = cacheState.Cache().Append(
		"attention",
		[]int{1, 12, 8, 64},
		make([]float64, 1*12*8*64),
		make([]float64, 1*12*8*64),
	)

	if err != nil {
		benchmark.Fatal(err)
	}

	for benchmark.Loop() {
		if _, err := cacheState.Snapshot(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}
