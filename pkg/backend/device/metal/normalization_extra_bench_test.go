package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func BenchmarkKernel_RunNorm3DDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalNormalizationDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			for _, name := range []string{"groupnorm", "instancenorm", "batchnorm_eval"} {
				name := name

				benchmark.Run(name, func(benchmark *testing.B) {
					run, out, byteCount, closeAll := norm3DBenchmarkSetup(
						benchmark, backend, storageDType, name,
					)
					defer closeAll()

					benchmark.SetBytes(byteCount)
					benchmark.ResetTimer()

					for benchmark.Loop() {
						if err := run(); err != nil {
							benchmark.Fatal(err)
						}

						if err := out.Sync(context.Background()); err != nil {
							benchmark.Fatal(err)
						}
					}
				})
			}
		})
	}
}

func norm3DBenchmarkSetup(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	name string,
) (func() error, tensor.Tensor, int64, func()) {
	batch, channels := norm3DShape()
	spatial := 8192
	fixture := norm3DFixtureForTest(batch, channels, spatial, storageDType)

	if name == "batchnorm_eval" {
		return batchNormEvalBenchmarkSetup(
			testingObject, backend, storageDType, batch, channels, spatial, fixture,
		)
	}

	input, scale, bias, out := norm3DAffineTensorsForTest(
		testingObject, backend, storageDType, batch, channels, spatial, fixture,
	)
	kernel := lookupNorm3DKernel(testingObject, name, storageDType)
	byteCount := int64(len(fixture.inputBytes)*2 + len(fixture.scaleBytes) + len(fixture.biasBytes))

	return func() error { return kernel.Run(input, scale, bias, out) },
		out,
		byteCount,
		func() { closeBenchmarkTensors(input, scale, bias, out) }
}

func batchNormEvalBenchmarkSetup(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	batch int,
	channels int,
	spatial int,
	fixture norm3DFixture,
) (func() error, tensor.Tensor, int64, func()) {
	input, scale, bias, mean, variance, out := batchNormEvalTensorsForTest(
		testingObject, backend, storageDType, batch, channels, spatial, fixture,
	)
	kernel := lookupBatchNormEvalKernel(testingObject, storageDType)
	byteCount := int64(
		len(fixture.inputBytes)*2 +
			len(fixture.scaleBytes) +
			len(fixture.biasBytes) +
			len(fixture.meanBytes) +
			len(fixture.varianceBytes),
	)

	return func() error { return kernel.Run(input, scale, bias, mean, variance, out) },
		out,
		byteCount,
		func() { closeBenchmarkTensors(input, scale, bias, mean, variance, out) }
}
