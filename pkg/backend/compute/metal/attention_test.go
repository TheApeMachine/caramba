//go:build darwin && cgo

package metal

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	cpuattention "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/attention"
	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/kv"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestMetalAttention_SDPATensor(test *testing.T) {
	lib := metallibPathOrSkip(test, "attention.metallib")

	Convey("Given resident Metal Q/K/V tensors", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		attentionOps, err := NewAttention(lib)
		So(err, ShouldBeNil)

		shape, err := computetensor.NewShape([]int{1, 1, 1, 2})
		So(err, ShouldBeNil)

		query := uploadMetalTensorForTest(test, tensorBackend, shape, []float64{1, 0})
		key := uploadMetalTensorForTest(test, tensorBackend, shape, []float64{1, 0})
		value := uploadMetalTensorForTest(test, tensorBackend, shape, []float64{3, 4})

		Convey("It should execute SDPA without host-backed inputs", func() {
			output, err := attentionOps.SDPATensor(
				query,
				key,
				value,
				shape,
				1,
				1,
				1,
				1,
				1,
				2,
				true,
			)

			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := tensorFloat64Values(output)

			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{3, 4})
		})

		Convey("It should match CPU causal SDPA for multi-token packed heads", func() {
			shape, err := computetensor.NewShape([]int{1, 2, 3, 4})
			So(err, ShouldBeNil)

			queryValues := []float64{
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1,
				1, 1, 0, 0,
				0, 1, 1, 0,
			}
			keyValues := []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
				1, 0, 1, 0,
				0, 1, 0, 1,
			}
			valueValues := []float64{
				3, 4, 5, 6,
				7, 8, 9, 10,
				11, 12, 13, 14,
				15, 16, 17, 18,
				19, 20, 21, 22,
				23, 24, 25, 26,
			}

			expectedState := state.NewDict().
				WithShape([]int{1, 2, 3, 4}).
				WithInputs(queryValues, keyValues, valueValues)
			expectedState.Causal = true
			expectedState, err = cpuattention.NewSDPA().Forward(expectedState)
			So(err, ShouldBeNil)

			output, err := attentionOps.SDPATensor(
				uploadMetalTensorForTest(test, tensorBackend, shape, queryValues),
				uploadMetalTensorForTest(test, tensorBackend, shape, keyValues),
				uploadMetalTensorForTest(test, tensorBackend, shape, valueValues),
				shape,
				1,
				2,
				3,
				3,
				3,
				4,
				true,
			)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := tensorFloat64Values(output)
			So(err, ShouldBeNil)
			assertAlmostEqualSlice(values, expectedState.Out, 1e-5)
		})

		Convey("It should match CPU causal SDPA for capacity-backed KV heads", func() {
			queryShape, err := computetensor.NewShape([]int{1, 2, 3, 4})
			So(err, ShouldBeNil)
			cacheShape, err := computetensor.NewShape([]int{1, 2, 5, 4})
			So(err, ShouldBeNil)

			queryValues := []float64{
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1,
				1, 1, 0, 0,
				0, 1, 1, 0,
			}
			liveKeyValues := []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
				1, 0, 1, 0,
				0, 1, 0, 1,
			}
			liveValueValues := []float64{
				3, 4, 5, 6,
				7, 8, 9, 10,
				11, 12, 13, 14,
				15, 16, 17, 18,
				19, 20, 21, 22,
				23, 24, 25, 26,
			}
			cacheKeyValues := []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				99, 99, 99, 99,
				100, 100, 100, 100,
				0, 0, 0, 4,
				1, 0, 1, 0,
				0, 1, 0, 1,
				99, 99, 99, 99,
				100, 100, 100, 100,
			}
			cacheValueValues := []float64{
				3, 4, 5, 6,
				7, 8, 9, 10,
				11, 12, 13, 14,
				99, 99, 99, 99,
				100, 100, 100, 100,
				15, 16, 17, 18,
				19, 20, 21, 22,
				23, 24, 25, 26,
				99, 99, 99, 99,
				100, 100, 100, 100,
			}

			expectedState := state.NewDict().
				WithShape([]int{1, 2, 3, 4}).
				WithInputs(queryValues, liveKeyValues, liveValueValues)
			expectedState.Causal = true
			expectedState, err = cpuattention.NewSDPA().Forward(expectedState)
			So(err, ShouldBeNil)

			output, err := attentionOps.SDPATensor(
				uploadMetalTensorForTest(test, tensorBackend, queryShape, queryValues),
				uploadMetalTensorForTest(test, tensorBackend, cacheShape, cacheKeyValues),
				uploadMetalTensorForTest(test, tensorBackend, cacheShape, cacheValueValues),
				queryShape,
				1,
				2,
				3,
				3,
				5,
				4,
				true,
			)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := tensorFloat64Values(output)
			So(err, ShouldBeNil)
			assertAlmostEqualSlice(values, expectedState.Out, 1e-5)
		})
	})
}

func TestMetalAttention_AppendKVTensor(test *testing.T) {
	lib := metallibPathOrSkip(test, "attention.metallib")

	Convey("Given resident Metal KV chunks", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		attentionOps, err := NewAttention(lib)
		So(err, ShouldBeNil)

		chunkShape, err := computetensor.NewShape([]int{1, 2, 1, 2})
		So(err, ShouldBeNil)

		firstKey := uploadMetalTensorForTest(test, tensorBackend, chunkShape, []float64{1, 2, 3, 4})
		firstValue := uploadMetalTensorForTest(test, tensorBackend, chunkShape, []float64{9, 10, 11, 12})

		Convey("It should append by token dimension for each head", func() {
			outputShape, err := computetensor.NewShape([]int{1, 2, 1, 2})
			So(err, ShouldBeNil)

			key, value, err := attentionOps.AppendKVTensor(
				nil,
				nil,
				firstKey,
				firstValue,
				outputShape,
				1,
				2,
				0,
				1,
				2,
			)
			So(err, ShouldBeNil)
			defer func() {
				So(key.Close(), ShouldBeNil)
				So(value.Close(), ShouldBeNil)
			}()

			nextKey := uploadMetalTensorForTest(test, tensorBackend, chunkShape, []float64{5, 6, 7, 8})
			nextValue := uploadMetalTensorForTest(test, tensorBackend, chunkShape, []float64{13, 14, 15, 16})

			outputShape, err = computetensor.NewShape([]int{1, 2, 2, 2})
			So(err, ShouldBeNil)

			appendedKey, appendedValue, err := attentionOps.AppendKVTensor(
				key,
				value,
				nextKey,
				nextValue,
				outputShape,
				1,
				2,
				1,
				1,
				2,
			)
			So(err, ShouldBeNil)
			defer func() {
				So(appendedKey.Close(), ShouldBeNil)
				So(appendedValue.Close(), ShouldBeNil)
			}()

			keyValues, err := tensorFloat64Values(appendedKey)
			So(err, ShouldBeNil)
			valueValues, err := tensorFloat64Values(appendedValue)
			So(err, ShouldBeNil)

			So(keyValues, ShouldResemble, []float64{1, 2, 5, 6, 3, 4, 7, 8})
			So(valueValues, ShouldResemble, []float64{9, 10, 13, 14, 11, 12, 15, 16})
		})
	})
}

func TestMetalAttention_GQATensor(test *testing.T) {
	lib := metallibPathOrSkip(test, "attention.metallib")

	Convey("Given resident Metal GQA tensors", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		attentionOps, err := NewAttention(lib)
		So(err, ShouldBeNil)

		queryShape, err := computetensor.NewShape([]int{1, 2, 1, 2})
		So(err, ShouldBeNil)

		keyValueShape, err := computetensor.NewShape([]int{1, 1, 2, 2})
		So(err, ShouldBeNil)

		query := []float64{0, 1, 1, 0}
		key := []float64{1, 0, 0, 1}
		value := []float64{2, 3, 5, 7}
		expectedState := state.NewDict().WithShape([]int{1, 2, 1, 2}).WithInputs(
			query,
			key,
			value,
		)
		expectedState.NumKVHeads = 1
		expectedState.HeadDim = 2
		expectedState.Causal = true
		expectedState, err = cpuattention.NewGQA().Forward(expectedState)
		So(err, ShouldBeNil)

		Convey("It should match CPU causal GQA with a longer KV history", func() {
			output, err := attentionOps.GQATensor(
				uploadMetalTensorForTest(test, tensorBackend, queryShape, query),
				uploadMetalTensorForTest(test, tensorBackend, keyValueShape, key),
				uploadMetalTensorForTest(test, tensorBackend, keyValueShape, value),
				queryShape,
				1,
				2,
				1,
				1,
				2,
				2,
				2,
				true,
			)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := tensorFloat64Values(output)
			So(err, ShouldBeNil)
			assertAlmostEqualSlice(values, expectedState.Out, 1e-5)
		})
	})
}

func TestTensorBackend_applySDPA(test *testing.T) {
	Convey("Given a Metal tensor backend and a generation KV cache", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		cache := kv.NewCache()
		So(cache.SetCapacity(8), ShouldBeNil)
		shape, err := computetensor.NewShape([]int{1, 1, 1, 2})
		So(err, ShouldBeNil)
		node := executor.NodeSpec{
			ID:    "attention",
			Op:    ir.OpType("attention.sdpa"),
			Shape: []int{1, 1, 1, 2},
			Metadata: map[string]any{
				"causal":   true,
				"kv_cache": cache,
			},
		}

		Convey("It should keep accumulated K/V tensors resident across decode steps", func() {
			output, err := tensorBackend.applySDPA(
				context.Background(),
				node,
				[]computetensor.Tensor{
					uploadMetalTensorForTest(test, tensorBackend, shape, []float64{1, 0}),
					uploadMetalTensorForTest(test, tensorBackend, shape, []float64{1, 0}),
					uploadMetalTensorForTest(test, tensorBackend, shape, []float64{3, 4}),
				},
			)
			So(err, ShouldBeNil)
			So(output.Close(), ShouldBeNil)
			So(tensorBackend.kvEntries["attention"].shape, ShouldResemble, []int{1, 1, 1, 2})
			So(tensorBackend.kvEntries["attention"].capacity, ShouldEqual, 8)

			output, err = tensorBackend.applySDPA(
				context.Background(),
				node,
				[]computetensor.Tensor{
					uploadMetalTensorForTest(test, tensorBackend, shape, []float64{0, 1}),
					uploadMetalTensorForTest(test, tensorBackend, shape, []float64{0, 1}),
					uploadMetalTensorForTest(test, tensorBackend, shape, []float64{5, 6}),
				},
			)
			So(err, ShouldBeNil)
			So(output.Close(), ShouldBeNil)
			So(tensorBackend.kvEntries["attention"].shape, ShouldResemble, []int{1, 1, 2, 2})
			So(tensorBackend.kvEntries["attention"].capacity, ShouldEqual, 8)

			cache.Reset()

			output, err = tensorBackend.applySDPA(
				context.Background(),
				node,
				[]computetensor.Tensor{
					uploadMetalTensorForTest(test, tensorBackend, shape, []float64{1, 0}),
					uploadMetalTensorForTest(test, tensorBackend, shape, []float64{1, 0}),
					uploadMetalTensorForTest(test, tensorBackend, shape, []float64{7, 8}),
				},
			)
			So(err, ShouldBeNil)
			So(output.Close(), ShouldBeNil)
			So(tensorBackend.kvEntries["attention"].shape, ShouldResemble, []int{1, 1, 1, 2})
			So(tensorBackend.kvEntries["attention"].capacity, ShouldEqual, 8)
		})

		Convey("It should grow cache capacity without losing resident tokens", func() {
			cache := kv.NewCache()
			node.Metadata["kv_cache"] = cache

			output, err := tensorBackend.applySDPA(
				context.Background(),
				node,
				[]computetensor.Tensor{
					uploadMetalTensorForTest(test, tensorBackend, shape, []float64{1, 0}),
					uploadMetalTensorForTest(test, tensorBackend, shape, []float64{1, 0}),
					uploadMetalTensorForTest(test, tensorBackend, shape, []float64{3, 4}),
				},
			)
			So(err, ShouldBeNil)
			So(output.Close(), ShouldBeNil)
			So(tensorBackend.kvEntries["attention"].capacity, ShouldEqual, 1)

			output, err = tensorBackend.applySDPA(
				context.Background(),
				node,
				[]computetensor.Tensor{
					uploadMetalTensorForTest(test, tensorBackend, shape, []float64{0, 1}),
					uploadMetalTensorForTest(test, tensorBackend, shape, []float64{0, 1}),
					uploadMetalTensorForTest(test, tensorBackend, shape, []float64{5, 6}),
				},
			)
			So(err, ShouldBeNil)
			So(output.Close(), ShouldBeNil)
			So(tensorBackend.kvEntries["attention"].shape, ShouldResemble, []int{1, 1, 2, 2})
			So(tensorBackend.kvEntries["attention"].capacity, ShouldEqual, 2)

			keyValues, err := tensorFloat64Values(tensorBackend.kvEntries["attention"].key)
			So(err, ShouldBeNil)
			So(keyValues, ShouldResemble, []float64{1, 0, 0, 1})
		})
	})
}

func TestTensorBackend_applyGQA(test *testing.T) {
	Convey("Given a Metal tensor backend and grouped-query KV cache", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		cache := kv.NewCache()
		So(cache.SetCapacity(8), ShouldBeNil)
		queryShape, err := computetensor.NewShape([]int{1, 2, 1, 2})
		So(err, ShouldBeNil)
		keyValueShape, err := computetensor.NewShape([]int{1, 1, 1, 2})
		So(err, ShouldBeNil)
		node := executor.NodeSpec{
			ID:    "gqa",
			Op:    ir.OpType("attention.gqa"),
			Shape: []int{1, 2, 1, 2},
			Metadata: map[string]any{
				"causal":       true,
				"num_kv_heads": 1,
				"kv_cache":     cache,
			},
		}

		Convey("It should keep grouped K/V tensors resident across decode steps", func() {
			output, err := tensorBackend.applyGQA(
				context.Background(),
				node,
				[]computetensor.Tensor{
					uploadMetalTensorForTest(test, tensorBackend, queryShape, []float64{1, 0, 0, 1}),
					uploadMetalTensorForTest(test, tensorBackend, keyValueShape, []float64{1, 0}),
					uploadMetalTensorForTest(test, tensorBackend, keyValueShape, []float64{2, 3}),
				},
			)
			So(err, ShouldBeNil)
			So(output.Close(), ShouldBeNil)
			So(tensorBackend.kvEntries["gqa"].shape, ShouldResemble, []int{1, 1, 1, 2})

			output, err = tensorBackend.applyGQA(
				context.Background(),
				node,
				[]computetensor.Tensor{
					uploadMetalTensorForTest(test, tensorBackend, queryShape, []float64{0, 1, 1, 0}),
					uploadMetalTensorForTest(test, tensorBackend, keyValueShape, []float64{0, 1}),
					uploadMetalTensorForTest(test, tensorBackend, keyValueShape, []float64{5, 7}),
				},
			)
			So(err, ShouldBeNil)
			So(output.Close(), ShouldBeNil)
			So(tensorBackend.kvEntries["gqa"].shape, ShouldResemble, []int{1, 1, 2, 2})
			So(tensorBackend.kvEntries["gqa"].capacity, ShouldEqual, 8)
		})
	})
}

func BenchmarkMetalAttention_SDPATensor(benchmark *testing.B) {
	lib := metallibPathOrSkip(benchmark, "attention.metallib")
	tensorBackend, err := NewTensorBackend()

	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}

	defer func() {
		_ = tensorBackend.Close()
	}()

	attentionOps, err := NewAttention(lib)

	if err != nil {
		benchmark.Fatal(err)
	}

	shape, err := computetensor.NewShape([]int{1, 12, 1, 64})

	if err != nil {
		benchmark.Fatal(err)
	}

	query := uploadMetalTensor(tensorBackend, shape, make([]float64, shape.Len()))
	key := uploadMetalTensor(tensorBackend, shape, make([]float64, shape.Len()))
	value := uploadMetalTensor(tensorBackend, shape, make([]float64, shape.Len()))
	defer func() {
		_ = query.Close()
		_ = key.Close()
		_ = value.Close()
	}()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		output, err := attentionOps.SDPATensor(query, key, value, shape, 1, 12, 1, 1, 1, 64, true)

		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkMetalAttention_AppendKVTensor(benchmark *testing.B) {
	lib := metallibPathOrSkip(benchmark, "attention.metallib")
	tensorBackend, err := NewTensorBackend()

	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}

	defer func() {
		_ = tensorBackend.Close()
	}()

	attentionOps, err := NewAttention(lib)

	if err != nil {
		benchmark.Fatal(err)
	}

	previousShape, err := computetensor.NewShape([]int{1, 12, 128, 64})

	if err != nil {
		benchmark.Fatal(err)
	}

	chunkShape, err := computetensor.NewShape([]int{1, 12, 1, 64})

	if err != nil {
		benchmark.Fatal(err)
	}

	outputShape, err := computetensor.NewShape([]int{1, 12, 129, 64})

	if err != nil {
		benchmark.Fatal(err)
	}

	previousKey := uploadMetalTensor(tensorBackend, previousShape, make([]float64, previousShape.Len()))
	previousValue := uploadMetalTensor(tensorBackend, previousShape, make([]float64, previousShape.Len()))
	keyChunk := uploadMetalTensor(tensorBackend, chunkShape, make([]float64, chunkShape.Len()))
	valueChunk := uploadMetalTensor(tensorBackend, chunkShape, make([]float64, chunkShape.Len()))
	defer func() {
		_ = previousKey.Close()
		_ = previousValue.Close()
		_ = keyChunk.Close()
		_ = valueChunk.Close()
	}()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		key, value, err := attentionOps.AppendKVTensor(
			previousKey,
			previousValue,
			keyChunk,
			valueChunk,
			outputShape,
			1,
			12,
			128,
			1,
			64,
		)

		if err != nil {
			benchmark.Fatal(err)
		}

		if err := key.Close(); err != nil {
			benchmark.Fatal(err)
		}

		if err := value.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func uploadMetalTensor(
	tensorBackend *TensorBackend,
	shape computetensor.Shape,
	values []float64,
) computetensor.Tensor {
	tensor, err := tensorBackend.UploadFloat64(shape, values)

	if err != nil {
		panic(err)
	}

	return tensor
}

func uploadMetalTensorForTest(
	test testing.TB,
	tensorBackend *TensorBackend,
	shape computetensor.Shape,
	values []float64,
) computetensor.Tensor {
	test.Helper()

	tensor := uploadMetalTensor(tensorBackend, shape, values)
	test.Cleanup(func() {
		_ = tensor.Close()
	})

	return tensor
}
