package kv

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestCache(t *testing.T) {
	Convey("Given an incremental KV cache", t, func() {
		cache := NewCache()

		Convey("Append should concatenate key/value chunks by attention node", func() {
			key, value, shape, err := cache.Append(
				"attention",
				[]int{1, 2, 1, 2},
				[]float64{1, 2, 3, 4},
				[]float64{5, 6, 7, 8},
			)
			So(err, ShouldBeNil)
			So(shape, ShouldResemble, []int{1, 2, 1, 2})
			So(key, ShouldResemble, []float64{1, 2, 3, 4})
			So(value, ShouldResemble, []float64{5, 6, 7, 8})

			key, value, shape, err = cache.Append(
				"attention",
				[]int{1, 2, 1, 2},
				[]float64{9, 10, 11, 12},
				[]float64{13, 14, 15, 16},
			)
			So(err, ShouldBeNil)
			So(shape, ShouldResemble, []int{1, 2, 2, 2})
			So(key, ShouldResemble, []float64{1, 2, 9, 10, 3, 4, 11, 12})
			So(value, ShouldResemble, []float64{5, 6, 13, 14, 7, 8, 15, 16})
		})

		Convey("Reset should clear accumulated state", func() {
			epoch := cache.Epoch()

			_, _, _, err := cache.Append(
				"attention",
				[]int{1, 1, 1, 1},
				[]float64{1},
				[]float64{2},
			)
			So(err, ShouldBeNil)

			cache.Reset()

			key, value, shape, err := cache.Append(
				"attention",
				[]int{1, 1, 1, 1},
				[]float64{3},
				[]float64{4},
			)
			So(err, ShouldBeNil)
			So(shape, ShouldResemble, []int{1, 1, 1, 1})
			So(key, ShouldResemble, []float64{3})
			So(value, ShouldResemble, []float64{4})
			So(cache.Epoch(), ShouldEqual, epoch+1)
		})

		Convey("Snapshot and Restore should preserve entries and capacity", func() {
			So(cache.SetCapacity(64), ShouldBeNil)
			_, _, _, err := cache.Append(
				"attention",
				[]int{1, 1, 2, 2},
				[]float64{1, 2, 3, 4},
				[]float64{5, 6, 7, 8},
			)
			So(err, ShouldBeNil)

			snapshot, err := cache.Snapshot()
			So(err, ShouldBeNil)
			So(snapshot.Capacity, ShouldEqual, 64)
			So(snapshot.Entries["attention"].Shape, ShouldResemble, []int{1, 1, 2, 2})

			restored := NewCache()
			So(restored.Restore(snapshot), ShouldBeNil)
			So(restored.Capacity(), ShouldEqual, 64)
			So(restored.EntryCount(), ShouldEqual, 1)

			restoredSnapshot, err := restored.Snapshot()
			So(err, ShouldBeNil)
			So(restoredSnapshot, ShouldResemble, snapshot)
		})
	})
}

func TestCache_Epoch(t *testing.T) {
	Convey("Given a KV cache", t, func() {
		cache := NewCache()

		Convey("It should expose a stable epoch until reset", func() {
			epoch := cache.Epoch()

			_, _, _, err := cache.Append(
				"attention",
				[]int{1, 1, 1, 1},
				[]float64{1},
				[]float64{2},
			)

			So(err, ShouldBeNil)
			So(cache.Epoch(), ShouldEqual, epoch)

			cache.Reset()

			So(cache.Epoch(), ShouldEqual, epoch+1)
		})
	})
}

func TestCache_SetCapacity(test *testing.T) {
	Convey("Given a KV cache", test, func() {
		cache := NewCache()

		Convey("It should store a backend resident token capacity", func() {
			err := cache.SetCapacity(128)

			So(err, ShouldBeNil)
			So(cache.Capacity(), ShouldEqual, 128)
		})

		Convey("It should reject negative capacity", func() {
			err := cache.SetCapacity(-1)

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "capacity")
		})
	})
}

func BenchmarkCache_Append(benchmark *testing.B) {
	cache := NewCache()
	shape := []int{1, 12, 1, 64}
	key := make([]float64, 1*12*1*64)
	value := make([]float64, 1*12*1*64)

	for benchmark.Loop() {
		_, _, _, _ = cache.Append("attention", shape, key, value)
	}
}

func BenchmarkCache_Epoch(benchmark *testing.B) {
	cache := NewCache()

	for benchmark.Loop() {
		_ = cache.Epoch()
	}
}

func BenchmarkCache_Capacity(benchmark *testing.B) {
	cache := NewCache()
	_ = cache.SetCapacity(1024)

	for benchmark.Loop() {
		_ = cache.Capacity()
	}
}

func BenchmarkCache_Snapshot(benchmark *testing.B) {
	cache := NewCache()
	shape := []int{1, 12, 8, 64}
	key := make([]float64, 1*12*8*64)
	value := make([]float64, 1*12*8*64)
	_, _, _, _ = cache.Append("attention", shape, key, value)

	for benchmark.Loop() {
		_, _ = cache.Snapshot()
	}
}
