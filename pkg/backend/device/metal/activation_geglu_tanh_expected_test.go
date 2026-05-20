package metal

import (
	cpuactivation "github.com/theapemachine/caramba/pkg/backend/device/cpu/activation"
	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
)

type geGLUTanhFixture struct {
	gateBytes     []byte
	upBytes       []byte
	expectedBytes []byte
}

func geGLUTanhFixtureForTest(elementCount int, storageDType dtype.DType) geGLUTanhFixture {
	gateValues := geGLUTanhGateValuesForTest(elementCount)
	upValues := geGLUTanhUpValuesForTest(elementCount)
	destination := make([]float32, elementCount)
	cpuactivation.GeGLUTanhTensorsF32Generic(
		&destination[0],
		&gateValues[0],
		&upValues[0],
		elementCount,
	)

	if storageDType == dtype.Float32 {
		return geGLUTanhFixture{
			gateBytes:     dtypeconvert.Float32ToBytes(gateValues),
			upBytes:       dtypeconvert.Float32ToBytes(upValues),
			expectedBytes: dtypeconvert.Float32ToBytes(destination),
		}
	}

	gateBytes := encodeFloat32ValuesAsDType(gateValues, storageDType)
	upBytes := encodeFloat32ValuesAsDType(upValues, storageDType)

	storedGate := decodeDTypeBytesToFloat32(gateBytes, storageDType)
	storedUp := decodeDTypeBytesToFloat32(upBytes, storageDType)
	roundTrip := make([]float32, elementCount)
	cpuactivation.GeGLUTanhTensorsF32Generic(
		&roundTrip[0],
		&storedGate[0],
		&storedUp[0],
		elementCount,
	)

	return geGLUTanhFixture{
		gateBytes:     gateBytes,
		upBytes:       upBytes,
		expectedBytes: encodeFloat32ValuesAsDType(roundTrip, storageDType),
	}
}

func geGLUTanhGateValuesForTest(elementCount int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = float32(index)*0.1 - 0.5
	}

	return values
}

func geGLUTanhUpValuesForTest(elementCount int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = float32(index)*0.1 + 0.5
	}

	return values
}
