//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "bridge_darwin.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"math/rand/v2"

	computekernels "github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

type metalSamplingConfig struct {
	logits       *metalTensor
	out          *metalTensor
	scores       *metalTensor
	indices      *metalTensor
	elementDType metalElementDType
	operation    metalSamplingOp
	count        uint32
	paddedCount  uint32
	target       float32
}

func runMetalSampling(
	operation metalSamplingOp,
	logits tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalSampling(operation, logits, out)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	if err := config.allocateScratch(); err != nil {
		return err
	}

	token, err := config.beginCompletion()
	if err != nil {
		config.closeScratch()
		return err
	}
	config.closeScratch()

	status := C.MetalStatus{}
	rc := C.metal_dispatch_sampling(
		config.logits.bridge.device,
		C.int(config.operation),
		C.int(config.elementDType),
		config.logits.buffer,
		config.scoreBuffer(),
		config.indexBuffer(),
		config.out.buffer,
		C.uint32_t(config.count),
		C.uint32_t(config.paddedCount),
		C.float(config.target),
		C.uint64_t(token),
		&status,
	)

	return finishMetalSamplingDispatch(token, rc, status)
}

func requireMetalSampling(
	operation metalSamplingOp,
	logits tensor.Tensor,
	out tensor.Tensor,
) (metalSamplingConfig, error) {
	logitTensor, outTensor, err := requireMetalSamplingTensors(logits, out)
	if err != nil {
		return metalSamplingConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(logitTensor.dtype)
	if err != nil {
		return metalSamplingConfig{}, err
	}

	count, paddedCount, err := metalSamplingCounts(logitTensor)
	if err != nil {
		return metalSamplingConfig{}, err
	}

	return metalSamplingConfig{
		logits: logitTensor, out: outTensor, elementDType: elementDType,
		operation: operation, count: count, paddedCount: paddedCount,
		target: metalSamplingDefaultTarget(),
	}, nil
}

func requireMetalSamplingTensors(logits tensor.Tensor, out tensor.Tensor) (*metalTensor, *metalTensor, error) {
	tensors, err := requireMetalTensors(logits, out)
	if err != nil {
		return nil, nil, err
	}

	logitTensor := tensors[0]
	outTensor := tensors[1]
	if outTensor.dtype != dtype.Int32 {
		return nil, nil, tensor.ErrDTypeMismatch
	}

	if logitTensor.bridge != outTensor.bridge {
		return nil, nil, errors.New("metal sampling: tensors belong to different Metal backends")
	}

	if outTensor.shape.Len() < 1 {
		return nil, nil, tensor.ErrShapeMismatch
	}

	return logitTensor, outTensor, nil
}

func metalSamplingCounts(logits *metalTensor) (uint32, uint32, error) {
	count := logits.shape.Len()
	if count <= 0 || count > 1<<31 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	paddedCount := uint32(1)
	for paddedCount < uint32(count) {
		paddedCount <<= 1
	}

	return uint32(count), paddedCount, nil
}

func (config *metalSamplingConfig) allocateScratch() error {
	if config.operation == metalSamplingGreedy {
		return nil
	}

	shape, err := tensor.NewShape([]int{int(config.paddedCount)})
	if err != nil {
		return err
	}

	if err := config.allocateScoreScratch(shape); err != nil {
		return err
	}

	return config.allocateIndexScratch(shape)
}

func (config *metalSamplingConfig) allocateScoreScratch(shape tensor.Shape) error {
	var err error
	config.scores, err = config.logits.bridge.empty(shape, dtype.Float32)

	return err
}

func (config *metalSamplingConfig) allocateIndexScratch(shape tensor.Shape) error {
	var err error
	config.indices, err = config.logits.bridge.empty(shape, dtype.Int32)
	if err == nil {
		return nil
	}

	_ = config.scores.Close()
	return err
}

func (config *metalSamplingConfig) beginCompletion() (uint64, error) {
	if config.operation == metalSamplingGreedy {
		return metalCompletions.Begin(config.out, config.logits)
	}

	return metalCompletions.BeginMany(
		[]*metalTensor{config.out, config.scores, config.indices},
		config.logits,
	)
}

func (config *metalSamplingConfig) closeScratch() {
	if config.scores != nil {
		_ = config.scores.Close()
	}

	if config.indices != nil {
		_ = config.indices.Close()
	}
}

func (config *metalSamplingConfig) scoreBuffer() C.MetalBufferRef {
	if config.scores == nil {
		return nil
	}

	return config.scores.buffer
}

func (config *metalSamplingConfig) indexBuffer() C.MetalBufferRef {
	if config.indices == nil {
		return nil
	}

	return config.indices.buffer
}

func finishMetalSamplingDispatch(token uint64, rc C.int, status C.MetalStatus) error {
	if rc == 0 {
		return nil
	}

	err := fmt.Errorf("metal sampling: %s", metalStatus("dispatch", status))
	metalCompletions.Fail(token, err)
	return err
}

func metalSamplingDefaultTarget() float32 {
	config := computekernels.DefaultSamplingConfig()
	source := rand.NewChaCha8([32]byte{
		byte(config.Seed), byte(config.Seed >> 8), byte(config.Seed >> 16), byte(config.Seed >> 24),
		byte(config.Seed >> 32), byte(config.Seed >> 40), byte(config.Seed >> 48), byte(config.Seed >> 56),
	})

	return rand.New(source).Float32()
}
