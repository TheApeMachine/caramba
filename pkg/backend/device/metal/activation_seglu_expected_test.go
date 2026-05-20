package metal

import (
	cpuactivation "github.com/theapemachine/caramba/pkg/backend/device/cpu/activation"
	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
)

type segluFixture struct {
	gateBytes     []byte
	upBytes       []byte
	expectedBytes []byte
}

func segluFixtureForTest(elementCount int, storageDType dtype.DType) segluFixture {
	gateValues := gluGateValuesForTest(elementCount)
	upValues := gluUpValuesForTest(elementCount)
	destination := make([]float32, elementCount)
	cpuactivation.SeGLUTensorsF32Generic(&destination[0], &gateValues[0], &upValues[0], elementCount)

	if storageDType == dtype.Float32 {
		return segluFixture{
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
	cpuactivation.SeGLUTensorsF32Generic(&roundTrip[0], &storedGate[0], &storedUp[0], elementCount)

	return segluFixture{
		gateBytes:     gateBytes,
		upBytes:       upBytes,
		expectedBytes: encodeFloat32ValuesAsDType(roundTrip, storageDType),
	}
}
