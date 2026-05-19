package hawkes

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func requireHawkesFloat32(format dtype.DType) {
	if format != dtype.Float32 {
		panic("hawkes: unsupported dtype")
	}
}

func HawkesIntensity(
	eventTimes, queryTimes, output unsafe.Pointer,
	eventCount, queryCount int,
	mu, alpha, beta float32,
	format dtype.DType,
) {
	requireHawkesFloat32(format)

	if queryCount == 0 {
		return
	}

	eventView := unsafe.Slice((*float32)(eventTimes), eventCount)
	queryView := unsafe.Slice((*float32)(queryTimes), queryCount)
	outputView := unsafe.Slice((*float32)(output), queryCount)

	HawkesIntensityNative(eventView, queryView, outputView, mu, alpha, beta)
}

func HawkesKernelMatrix(
	eventTimes, output unsafe.Pointer,
	eventCount int,
	alpha, beta float32,
	format dtype.DType,
) {
	requireHawkesFloat32(format)

	if eventCount == 0 {
		return
	}

	eventView := unsafe.Slice((*float32)(eventTimes), eventCount)
	outputView := unsafe.Slice((*float32)(output), eventCount*eventCount)

	HawkesKernelMatrixNative(eventView, outputView, alpha, beta)
}

func HawkesLogLikelihood(
	eventTimes unsafe.Pointer,
	eventCount int,
	totalT, mu, alpha, beta float32,
	output unsafe.Pointer,
	format dtype.DType,
) {
	requireHawkesFloat32(format)

	if eventCount == 0 {
		return
	}

	eventView := unsafe.Slice((*float32)(eventTimes), eventCount)
	outputView := unsafe.Slice((*float32)(output), 1)

	HawkesLogLikelihoodNative(eventView, totalT, mu, alpha, beta, outputView)
}

func MarkovMutualInformation(
	joint, output unsafe.Pointer,
	xCount, yCount int,
	format dtype.DType,
) {
	requireHawkesFloat32(format)

	if xCount == 0 || yCount == 0 {
		return
	}

	jointView := unsafe.Slice((*float32)(joint), xCount*yCount)
	outputView := unsafe.Slice((*float32)(output), 1)

	MarkovMutualInformationNative(jointView, xCount, yCount, outputView)
}

func MarkovBlanketPartition(
	adjacency, internal, output unsafe.Pointer,
	nodeCount, internalCount int,
	format dtype.DType,
) {
	requireHawkesFloat32(format)

	if nodeCount == 0 {
		return
	}

	adjacencyView := unsafe.Slice((*float32)(adjacency), nodeCount*nodeCount)
	internalView := unsafe.Slice((*int32)(internal), internalCount)
	outputView := unsafe.Slice((*int32)(output), nodeCount)

	markovBlanketPartition(adjacencyView, internalView, outputView, nodeCount)
}
