package metal

import (
	cpuhawkes "github.com/theapemachine/caramba/pkg/backend/device/cpu/hawkes"
	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
)

const hawkesLogLikelihoodMaxULP uint32 = 4

type hawkesLogLikelihoodFixture struct {
	eventBytes     []byte
	totalTimeBytes []byte
	baselineBytes  []byte
	alphaBytes     []byte
	betaBytes      []byte
	expectedBytes  []byte
}

func hawkesLogLikelihoodDTypeBytes(elementCount int, storageDType dtype.DType) hawkesLogLikelihoodFixture {
	events := hawkesEventTimes(elementCount)
	totalTime := events[len(events)-1] + 1.0
	mu, alpha, beta := float32(0.25), float32(0.125), float32(0.375)

	if storageDType == dtype.Float32 {
		expected := make([]float32, 1)
		cpuhawkes.HawkesLogLikelihoodScalar(events, totalTime, mu, alpha, beta, expected)

		return hawkesLogLikelihoodFixture{
			eventBytes:     dtypeconvert.Float32ToBytes(events),
			totalTimeBytes: dtypeconvert.Float32ToBytes([]float32{totalTime}),
			baselineBytes:  dtypeconvert.Float32ToBytes([]float32{mu}),
			alphaBytes:     dtypeconvert.Float32ToBytes([]float32{alpha}),
			betaBytes:      dtypeconvert.Float32ToBytes([]float32{beta}),
			expectedBytes:  dtypeconvert.Float32ToBytes(expected),
		}
	}

	eventBytes := encodeResearchValuesAsDType(events, storageDType)
	totalTimeBytes := encodeResearchValuesAsDType([]float32{totalTime}, storageDType)
	baselineBytes := encodeResearchValuesAsDType([]float32{mu}, storageDType)
	alphaBytes := encodeResearchValuesAsDType([]float32{alpha}, storageDType)
	betaBytes := encodeResearchValuesAsDType([]float32{beta}, storageDType)

	storedEvents := decodeDTypeBytesToFloat32(eventBytes, storageDType)
	storedTotalTime := decodeDTypeBytesToFloat32(totalTimeBytes, storageDType)[0]
	storedMu := decodeDTypeBytesToFloat32(baselineBytes, storageDType)[0]
	storedAlpha := decodeDTypeBytesToFloat32(alphaBytes, storageDType)[0]
	storedBeta := decodeDTypeBytesToFloat32(betaBytes, storageDType)[0]

	expected := make([]float32, 1)
	cpuhawkes.HawkesLogLikelihoodScalar(
		storedEvents, storedTotalTime, storedMu, storedAlpha, storedBeta, expected,
	)

	return hawkesLogLikelihoodFixture{
		eventBytes:     eventBytes,
		totalTimeBytes: totalTimeBytes,
		baselineBytes:  baselineBytes,
		alphaBytes:     alphaBytes,
		betaBytes:      betaBytes,
		expectedBytes:  encodeResearchValuesAsDType(expected, storageDType),
	}
}
