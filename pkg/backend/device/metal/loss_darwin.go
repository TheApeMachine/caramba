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
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

type metalPairLossConfig struct {
	predictions  *metalTensor
	targets      *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	count        uint32
}

type metalCrossEntropyLossConfig struct {
	logits       *metalTensor
	targets      *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	batch        uint32
	classes      uint32
}

const metalLossThreadCountGo = 256

func runMetalPairLossKernel(operation metalLossOp, args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalPairLoss(operation, args[0], args[1], args[2])
}

func runMetalPairLoss(
	operation metalLossOp,
	predictions tensor.Tensor,
	targets tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalPairLoss(predictions, targets, out)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	partialCount := metalLossPartialCount(int(config.count))
	scratch, err := newMetalLossScratch(config.out.bridge, partialCount)
	if err != nil {
		return err
	}

	token, err := metalCompletions.BeginMany(
		[]*metalTensor{config.out, scratch},
		config.predictions,
		config.targets,
	)
	if err != nil {
		_ = scratch.Close()
		return err
	}

	_ = scratch.Close()

	status := C.MetalStatus{}
	rc := C.metal_dispatch_pair_loss(
		config.predictions.bridge.device,
		C.int(operation),
		C.int(config.elementDType),
		config.predictions.buffer,
		config.targets.buffer,
		scratch.buffer,
		config.out.buffer,
		C.uint32_t(config.count),
		C.uint32_t(partialCount),
		C.uint64_t(token),
		&status,
	)

	return finishMetalLossDispatch("pair_loss", token, rc, status)
}

func runMetalCrossEntropyLoss(
	logits tensor.Tensor,
	targets tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalCrossEntropyLoss(logits, targets, out)
	if err != nil {
		return err
	}

	if config.batch == 0 {
		return nil
	}

	scratch, err := newMetalLossScratch(config.out.bridge, int(config.batch))
	if err != nil {
		return err
	}

	token, err := metalCompletions.BeginMany(
		[]*metalTensor{config.out, scratch},
		config.logits,
		config.targets,
	)
	if err != nil {
		_ = scratch.Close()
		return err
	}

	_ = scratch.Close()

	status := C.MetalStatus{}
	rc := C.metal_dispatch_cross_entropy_loss(
		config.logits.bridge.device,
		C.int(config.elementDType),
		config.logits.buffer,
		config.targets.buffer,
		scratch.buffer,
		config.out.buffer,
		C.uint32_t(config.batch),
		C.uint32_t(config.classes),
		C.uint64_t(token),
		&status,
	)

	return finishMetalLossDispatch("cross_entropy", token, rc, status)
}

func metalLossPartialCount(elementCount int) int {
	return (elementCount + metalLossThreadCountGo - 1) / metalLossThreadCountGo
}

func newMetalLossScratch(bridge *metalBridge, partialCount int) (*metalTensor, error) {
	shape, err := tensor.NewShape([]int{partialCount})
	if err != nil {
		return nil, err
	}

	return bridge.empty(shape, dtype.Float32)
}

func requireMetalPairLoss(
	predictions tensor.Tensor,
	targets tensor.Tensor,
	out tensor.Tensor,
) (metalPairLossConfig, error) {
	predictionTensor, targetTensor, outTensor, err := requireMetalSameDType3(
		predictions, targets, out,
	)
	if err != nil {
		return metalPairLossConfig{}, err
	}

	if !predictionTensor.shape.Equal(targetTensor.shape) || outTensor.shape.Len() < 1 {
		return metalPairLossConfig{}, tensor.ErrShapeMismatch
	}

	if predictionTensor.shape.Len() > math.MaxUint32 {
		return metalPairLossConfig{}, tensor.ErrShapeMismatch
	}

	elementDType, err := metalElementDTypeFor(predictionTensor.dtype)
	if err != nil {
		return metalPairLossConfig{}, err
	}

	return metalPairLossConfig{
		predictions:  predictionTensor,
		targets:      targetTensor,
		out:          outTensor,
		elementDType: elementDType,
		count:        uint32(predictionTensor.shape.Len()),
	}, nil
}

func requireMetalCrossEntropyLoss(
	logits tensor.Tensor,
	targets tensor.Tensor,
	out tensor.Tensor,
) (metalCrossEntropyLossConfig, error) {
	tensors, err := requireMetalTensors(logits, targets, out)
	if err != nil {
		return metalCrossEntropyLossConfig{}, err
	}

	if tensors[1].dtype != dtype.Int32 || tensors[0].dtype != tensors[2].dtype {
		return metalCrossEntropyLossConfig{}, tensor.ErrDTypeMismatch
	}

	if tensors[0].bridge != tensors[1].bridge || tensors[0].bridge != tensors[2].bridge {
		return metalCrossEntropyLossConfig{}, errors.New("metal loss: tensors belong to different Metal backends")
	}

	batch, classes, err := metalCrossEntropyDims(tensors[0], tensors[1], tensors[2])
	if err != nil {
		return metalCrossEntropyLossConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalCrossEntropyLossConfig{}, err
	}

	return metalCrossEntropyLossConfig{
		logits: tensors[0], targets: tensors[1], out: tensors[2],
		elementDType: elementDType, batch: uint32(batch), classes: uint32(classes),
	}, nil
}

func metalCrossEntropyDims(logits *metalTensor, targets *metalTensor, out *metalTensor) (int, int, error) {
	dims := logits.shape.Dims()
	if len(dims) == 0 || out.shape.Len() < 1 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	classes := dims[len(dims)-1]
	if classes == 0 || logits.shape.Len()%classes != 0 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	batch := logits.shape.Len() / classes
	if targets.shape.Len() != batch || batch > math.MaxUint32 || classes > math.MaxUint32 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	return batch, classes, nil
}

func finishMetalLossDispatch(name string, token uint64, rc C.int, status C.MetalStatus) error {
	if rc == 0 {
		return nil
	}

	err := fmt.Errorf("metal %s: %s", name, metalStatus("dispatch", status))
	metalCompletions.Fail(token, err)
	return err
}
