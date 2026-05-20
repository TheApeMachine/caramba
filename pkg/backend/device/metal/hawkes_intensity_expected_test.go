package metal

import (
	cpuhawkes "github.com/theapemachine/caramba/pkg/backend/device/cpu/hawkes"
	"github.com/theapemachine/manifesto/dtype"
	dtypeconvert "github.com/theapemachine/manifesto/dtype/convert"
)

const hawkesIntensityMaxULP uint32 = 4

type hawkesIntensityFixture struct {
	eventBytes    []byte
	queryBytes    []byte
	baselineBytes []byte
	alphaBytes    []byte
	betaBytes     []byte
	expectedBytes []byte
}

func hawkesIntensityEventTimesForTest(elementCount int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = float32(index) / float32(elementCount)
	}

	return values
}

func hawkesIntensityQueryTimesForTest(elementCount int) []float32 {
	values := make([]float32, elementCount)

	for index := range values {
		values[index] = float32(index+1) / float32(elementCount)
	}

	return values
}

func hawkesIntensityDTypeBytes(elementCount int, storageDType dtype.DType) hawkesIntensityFixture {
	events := hawkesIntensityEventTimesForTest(elementCount)
	queries := hawkesIntensityQueryTimesForTest(elementCount)
	mu, alpha, beta := float32(0.1), float32(0.5), float32(1.0)

	if storageDType == dtype.Float32 {
		expected := make([]float32, elementCount)
		cpuhawkes.HawkesIntensityScalar(events, queries, expected, mu, alpha, beta)

		return hawkesIntensityFixture{
			eventBytes:    dtypeconvert.Float32ToBytes(events),
			queryBytes:    dtypeconvert.Float32ToBytes(queries),
			baselineBytes: dtypeconvert.Float32ToBytes([]float32{mu}),
			alphaBytes:    dtypeconvert.Float32ToBytes([]float32{alpha}),
			betaBytes:     dtypeconvert.Float32ToBytes([]float32{beta}),
			expectedBytes: dtypeconvert.Float32ToBytes(expected),
		}
	}

	eventBytes := encodeResearchValuesAsDType(events, storageDType)
	queryBytes := encodeResearchValuesAsDType(queries, storageDType)
	baselineBytes := encodeResearchValuesAsDType([]float32{mu}, storageDType)
	alphaBytes := encodeResearchValuesAsDType([]float32{alpha}, storageDType)
	betaBytes := encodeResearchValuesAsDType([]float32{beta}, storageDType)

	storedEvents := decodeDTypeBytesToFloat32(eventBytes, storageDType)
	storedQueries := decodeDTypeBytesToFloat32(queryBytes, storageDType)
	storedMu := decodeDTypeBytesToFloat32(baselineBytes, storageDType)[0]
	storedAlpha := decodeDTypeBytesToFloat32(alphaBytes, storageDType)[0]
	storedBeta := decodeDTypeBytesToFloat32(betaBytes, storageDType)[0]

	expected := make([]float32, elementCount)
	cpuhawkes.HawkesIntensityScalar(
		storedEvents, storedQueries, expected, storedMu, storedAlpha, storedBeta,
	)

	return hawkesIntensityFixture{
		eventBytes:    eventBytes,
		queryBytes:    queryBytes,
		baselineBytes: baselineBytes,
		alphaBytes:    alphaBytes,
		betaBytes:     betaBytes,
		expectedBytes: encodeResearchValuesAsDType(expected, storageDType),
	}
}
