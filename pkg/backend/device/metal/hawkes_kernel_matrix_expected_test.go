package metal

import (
	cpuhawkes "github.com/theapemachine/caramba/pkg/backend/device/cpu/hawkes"
	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
)

const hawkesKernelMatrixMaxULP uint32 = 4

type hawkesKernelMatrixFixture struct {
	eventBytes    []byte
	alphaBytes    []byte
	betaBytes     []byte
	expectedBytes []byte
}

func hawkesKernelMatrixDTypeBytes(elementCount int, storageDType dtype.DType) hawkesKernelMatrixFixture {
	eventCount := hawkesMatrixEventCount(elementCount)
	events := hawkesEventTimes(eventCount)
	alpha, beta := float32(0.5), float32(1.0)

	if storageDType == dtype.Float32 {
		expected := make([]float32, eventCount*eventCount)
		cpuhawkes.HawkesKernelMatrixScalar(events, expected, alpha, beta)

		return hawkesKernelMatrixFixture{
			eventBytes:    dtypeconvert.Float32ToBytes(events),
			alphaBytes:    dtypeconvert.Float32ToBytes([]float32{alpha}),
			betaBytes:     dtypeconvert.Float32ToBytes([]float32{beta}),
			expectedBytes: dtypeconvert.Float32ToBytes(expected),
		}
	}

	eventBytes := encodeResearchValuesAsDType(events, storageDType)
	alphaBytes := encodeResearchValuesAsDType([]float32{alpha}, storageDType)
	betaBytes := encodeResearchValuesAsDType([]float32{beta}, storageDType)
	storedEvents := decodeDTypeBytesToFloat32(eventBytes, storageDType)
	storedAlpha := decodeDTypeBytesToFloat32(alphaBytes, storageDType)[0]
	storedBeta := decodeDTypeBytesToFloat32(betaBytes, storageDType)[0]

	expected := make([]float32, eventCount*eventCount)
	cpuhawkes.HawkesKernelMatrixScalar(storedEvents, expected, storedAlpha, storedBeta)

	return hawkesKernelMatrixFixture{
		eventBytes:    eventBytes,
		alphaBytes:    alphaBytes,
		betaBytes:     betaBytes,
		expectedBytes: encodeResearchValuesAsDType(expected, storageDType),
	}
}
