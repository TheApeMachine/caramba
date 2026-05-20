package checkpoint

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestRunCheckpointEncodeFloat32ParityLengths(t *testing.T) {
	convey.Convey("Given RunCheckpointEncodeFloat32", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should round-trip scalar decode for N=%d", length), func() {
				inputValues := randomFloat32Vector(length, 0x2920+int64(length))
				dims := []int{length}

				inputTensor := uploadHostFloat32Tensor(t, inputValues)
				defer inputTensor.Close()

				headerBytes := 16 + len(dims)*8
				dataBytes := length * 4
				encodedTensor := newHostUint8Tensor(t, headerBytes+dataBytes)
				defer encodedTensor.Close()

				encodeErr := RunCheckpointEncodeFloat32(inputTensor, encodedTensor)
				convey.So(encodeErr, convey.ShouldBeNil)

				decodedTensor := uploadHostFloat32Tensor(t, make([]float32, length))
				defer decodedTensor.Close()

				decodeErr := RunCheckpointDecodeFloat32(encodedTensor, decodedTensor)
				convey.So(decodeErr, convey.ShouldBeNil)

				decodedValues, nativeErr := decodedTensor.Float32Native()
				convey.So(nativeErr, convey.ShouldBeNil)

				assertFloat32SliceEqual(t, decodedValues, inputValues)
			})
		}
	})
}

func uploadHostFloat32Tensor(
	testingTB interface {
		Helper()
		Fatalf(string, ...any)
	},
	values []float32,
) tensor.Tensor {
	testingTB.Helper()

	backend := tensor.NewHostBackend()
	shape, shapeErr := tensor.NewShape([]int{len(values)})

	if shapeErr != nil {
		testingTB.Fatalf("NewShape: %v", shapeErr)
	}

	bytesIn := float32SliceToBytes(values)
	uploaded, uploadErr := backend.Upload(shape, dtype.Float32, bytesIn)

	if uploadErr != nil {
		testingTB.Fatalf("Upload: %v", uploadErr)
	}

	return uploaded
}

func newHostUint8Tensor(
	testingTB interface {
		Helper()
		Fatalf(string, ...any)
	},
	byteCount int,
) tensor.Tensor {
	testingTB.Helper()

	shape, shapeErr := tensor.NewShape([]int{byteCount})

	if shapeErr != nil {
		testingTB.Fatalf("NewShape: %v", shapeErr)
	}

	created, createErr := tensor.New(shape, dtype.Uint8)

	if createErr != nil {
		testingTB.Fatalf("New: %v", createErr)
	}

	return created
}

func float32SliceToBytes(values []float32) []byte {
	bytesOut := make([]byte, len(values)*4)

	for index, value := range values {
		binary.LittleEndian.PutUint32(bytesOut[index*4:], math.Float32bits(value))
	}

	return bytesOut
}
