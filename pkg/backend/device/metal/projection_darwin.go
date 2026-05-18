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

type metalLinearConfig struct {
	input        *metalTensor
	weight       *metalTensor
	bias         *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	batch        uint32
	inner        uint32
	outDim       uint32
}

type metalFusedQKVConfig struct {
	input        *metalTensor
	weight       *metalTensor
	bias         *metalTensor
	query        *metalTensor
	key          *metalTensor
	value        *metalTensor
	elementDType metalElementDType
	batch        uint32
	inner        uint32
	outDim       uint32
}

type metalLoRAMergeConfig struct {
	baseWeight   *metalTensor
	loraA        *metalTensor
	loraB        *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	outDim       uint32
	rank         uint32
	inner        uint32
}

type metalLoRAApplyConfig struct {
	baseOut      *metalTensor
	loraA        *metalTensor
	loraB        *metalTensor
	input        *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	batch        uint32
	inner        uint32
	rank         uint32
	outDim       uint32
}

func runMetalLinear(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalLinear(input, weight, bias, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.input, config.weight, config.bias)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_linear(
		config.input.bridge.device,
		C.int(config.elementDType),
		config.input.buffer,
		config.weight.buffer,
		config.bias.buffer,
		config.out.buffer,
		C.uint32_t(config.batch),
		C.uint32_t(config.inner),
		C.uint32_t(config.outDim),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal linear: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func runMetalFusedQKV(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	query tensor.Tensor,
	key tensor.Tensor,
	value tensor.Tensor,
) error {
	config, err := requireMetalFusedQKV(input, weight, bias, query, key, value)
	if err != nil {
		return err
	}

	if config.query.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.BeginMany(
		[]*metalTensor{config.query, config.key, config.value},
		config.input, config.weight, config.bias,
	)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_fused_qkv(
		config.input.bridge.device,
		C.int(config.elementDType),
		config.input.buffer,
		config.weight.buffer,
		config.bias.buffer,
		config.query.buffer,
		config.key.buffer,
		config.value.buffer,
		C.uint32_t(config.batch),
		C.uint32_t(config.inner),
		C.uint32_t(config.outDim),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal fused_qkv: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func runMetalLoRAMerge(
	baseWeight tensor.Tensor,
	loraA tensor.Tensor,
	loraB tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalLoRAMerge(baseWeight, loraA, loraB, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.baseWeight, config.loraA, config.loraB)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_lora_merge(
		config.baseWeight.bridge.device,
		C.int(config.elementDType),
		config.baseWeight.buffer,
		config.loraA.buffer,
		config.loraB.buffer,
		config.out.buffer,
		C.uint32_t(config.outDim),
		C.uint32_t(config.rank),
		C.uint32_t(config.inner),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal lora_merge: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func runMetalLoRAApply(
	baseOut tensor.Tensor,
	loraA tensor.Tensor,
	loraB tensor.Tensor,
	input tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalLoRAApply(baseOut, loraA, loraB, input, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	scratch, err := loraApplyScratch(config)
	if err != nil {
		return err
	}
	defer func() {
		_ = scratch.Close()
	}()

	token, err := metalCompletions.BeginMany(
		[]*metalTensor{config.out, scratch},
		config.baseOut, config.loraA, config.loraB, config.input,
	)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_lora_apply(
		config.baseOut.bridge.device,
		C.int(config.elementDType),
		config.baseOut.buffer,
		config.loraA.buffer,
		config.loraB.buffer,
		config.input.buffer,
		scratch.buffer,
		config.out.buffer,
		C.uint32_t(config.batch),
		C.uint32_t(config.inner),
		C.uint32_t(config.rank),
		C.uint32_t(config.outDim),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal lora_apply: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func requireMetalLinear(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) (metalLinearConfig, error) {
	config, err := metalLinearTensors(input, weight, bias, out)
	if err != nil {
		return metalLinearConfig{}, err
	}

	batch, inner, outDim, err := metalLinearDims(config)
	if err != nil {
		return metalLinearConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(config.input.dtype)
	if err != nil {
		return metalLinearConfig{}, err
	}

	config.batch = uint32(batch)
	config.inner = uint32(inner)
	config.outDim = uint32(outDim)
	config.elementDType = elementDType
	return config, nil
}

func requireMetalFusedQKV(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	query tensor.Tensor,
	key tensor.Tensor,
	value tensor.Tensor,
) (metalFusedQKVConfig, error) {
	config, err := metalFusedQKVTensors(input, weight, bias, query, key, value)
	if err != nil {
		return metalFusedQKVConfig{}, err
	}

	batch, inner, outDim, err := metalFusedQKVDims(config)
	if err != nil {
		return metalFusedQKVConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(config.input.dtype)
	if err != nil {
		return metalFusedQKVConfig{}, err
	}

	config.batch = uint32(batch)
	config.inner = uint32(inner)
	config.outDim = uint32(outDim)
	config.elementDType = elementDType
	return config, nil
}

func requireMetalLoRAMerge(
	baseWeight tensor.Tensor,
	loraA tensor.Tensor,
	loraB tensor.Tensor,
	out tensor.Tensor,
) (metalLoRAMergeConfig, error) {
	config, err := metalLoRAMergeTensors(baseWeight, loraA, loraB, out)
	if err != nil {
		return metalLoRAMergeConfig{}, err
	}

	outDim, rank, inner, err := metalLoRAMergeDims(config)
	if err != nil {
		return metalLoRAMergeConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(config.baseWeight.dtype)
	if err != nil {
		return metalLoRAMergeConfig{}, err
	}

	config.outDim = uint32(outDim)
	config.rank = uint32(rank)
	config.inner = uint32(inner)
	config.elementDType = elementDType
	return config, nil
}

func requireMetalLoRAApply(
	baseOut tensor.Tensor,
	loraA tensor.Tensor,
	loraB tensor.Tensor,
	input tensor.Tensor,
	out tensor.Tensor,
) (metalLoRAApplyConfig, error) {
	config, err := metalLoRAApplyTensors(baseOut, loraA, loraB, input, out)
	if err != nil {
		return metalLoRAApplyConfig{}, err
	}

	batch, inner, rank, outDim, err := metalLoRAApplyDims(config)
	if err != nil {
		return metalLoRAApplyConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(config.baseOut.dtype)
	if err != nil {
		return metalLoRAApplyConfig{}, err
	}

	config.batch = uint32(batch)
	config.inner = uint32(inner)
	config.rank = uint32(rank)
	config.outDim = uint32(outDim)
	config.elementDType = elementDType
	return config, nil
}

func loraApplyScratch(config metalLoRAApplyConfig) (*metalTensor, error) {
	shape, err := tensor.NewShape([]int{int(config.batch), int(config.rank)})
	if err != nil {
		return nil, err
	}

	target, err := config.baseOut.bridge.empty(shape, dtype.Float32)
	if err != nil {
		return nil, err
	}

	return target, nil
}

func metalLinearTensors(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) (metalLinearConfig, error) {
	inputTensor, weightTensor, biasTensor, outTensor, err := requireMetalTensors4(
		input, weight, bias, out,
	)
	if err != nil {
		return metalLinearConfig{}, err
	}

	config := metalLinearConfig{
		input: inputTensor, weight: weightTensor, bias: biasTensor, out: outTensor,
	}
	return config, requireMetalProjectionSameDTypeAndBridge(
		inputTensor, weightTensor, biasTensor, outTensor,
	)
}

func metalFusedQKVTensors(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	query tensor.Tensor,
	key tensor.Tensor,
	value tensor.Tensor,
) (metalFusedQKVConfig, error) {
	tensors, err := requireMetalTensors(input, weight, bias, query, key, value)
	if err != nil {
		return metalFusedQKVConfig{}, err
	}

	config := metalFusedQKVConfig{
		input: tensors[0], weight: tensors[1], bias: tensors[2],
		query: tensors[3], key: tensors[4], value: tensors[5],
	}
	return config, requireMetalProjectionSameDTypeAndBridge(tensors...)
}

func metalLoRAMergeTensors(
	baseWeight tensor.Tensor,
	loraA tensor.Tensor,
	loraB tensor.Tensor,
	out tensor.Tensor,
) (metalLoRAMergeConfig, error) {
	baseTensor, loraATensor, loraBTensor, outTensor, err := requireMetalTensors4(
		baseWeight, loraA, loraB, out,
	)
	if err != nil {
		return metalLoRAMergeConfig{}, err
	}

	config := metalLoRAMergeConfig{
		baseWeight: baseTensor, loraA: loraATensor, loraB: loraBTensor, out: outTensor,
	}
	return config, requireMetalProjectionSameDTypeAndBridge(
		baseTensor, loraATensor, loraBTensor, outTensor,
	)
}

func metalLoRAApplyTensors(
	baseOut tensor.Tensor,
	loraA tensor.Tensor,
	loraB tensor.Tensor,
	input tensor.Tensor,
	out tensor.Tensor,
) (metalLoRAApplyConfig, error) {
	tensors, err := requireMetalTensors(baseOut, loraA, loraB, input, out)
	if err != nil {
		return metalLoRAApplyConfig{}, err
	}

	config := metalLoRAApplyConfig{
		baseOut: tensors[0], loraA: tensors[1], loraB: tensors[2],
		input: tensors[3], out: tensors[4],
	}
	return config, requireMetalProjectionSameDTypeAndBridge(tensors...)
}

func metalLinearDims(config metalLinearConfig) (int, int, int, error) {
	inputDims := config.input.shape.Dims()
	weightDims := config.weight.shape.Dims()
	biasDims := config.bias.shape.Dims()
	outDims := config.out.shape.Dims()

	if len(inputDims) != 2 || len(weightDims) != 2 ||
		len(biasDims) != 1 || len(outDims) != 2 {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	batch, inner, outDim := inputDims[0], inputDims[1], weightDims[0]
	if weightDims[1] != inner || biasDims[0] != outDim ||
		outDims[0] != batch || outDims[1] != outDim {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	return batch, inner, outDim, requireProjectionUint32(batch, inner, outDim)
}

func metalFusedQKVDims(config metalFusedQKVConfig) (int, int, int, error) {
	inputDims := config.input.shape.Dims()
	weightDims := config.weight.shape.Dims()
	biasDims := config.bias.shape.Dims()
	queryDims := config.query.shape.Dims()

	if len(inputDims) != 2 || len(weightDims) != 2 ||
		len(biasDims) != 1 || len(queryDims) != 2 {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	batch, inner, fusedOut := inputDims[0], inputDims[1], weightDims[0]
	if fusedOut%3 != 0 || weightDims[1] != inner || biasDims[0] != fusedOut {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	outDim := fusedOut / 3
	if err := requireFusedQKVOutputDims(config, batch, outDim); err != nil {
		return 0, 0, 0, err
	}

	return batch, inner, outDim, requireProjectionUint32(batch, inner, outDim)
}

func metalLoRAMergeDims(config metalLoRAMergeConfig) (int, int, int, error) {
	baseDims := config.baseWeight.shape.Dims()
	loraADims := config.loraA.shape.Dims()
	loraBDims := config.loraB.shape.Dims()
	outDims := config.out.shape.Dims()

	if len(baseDims) != 2 || len(loraADims) != 2 ||
		len(loraBDims) != 2 || len(outDims) != 2 {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	outDim, rank, inner := loraADims[0], loraADims[1], loraBDims[1]
	if loraBDims[0] != rank || baseDims[0] != outDim ||
		baseDims[1] != inner || outDims[0] != outDim || outDims[1] != inner {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	return outDim, rank, inner, requireProjectionUint32(outDim, rank, inner)
}

func metalLoRAApplyDims(config metalLoRAApplyConfig) (int, int, int, int, error) {
	baseDims := config.baseOut.shape.Dims()
	loraADims := config.loraA.shape.Dims()
	loraBDims := config.loraB.shape.Dims()
	inputDims := config.input.shape.Dims()
	outDims := config.out.shape.Dims()

	if len(baseDims) != 2 || len(loraADims) != 2 ||
		len(loraBDims) != 2 || len(inputDims) != 2 || len(outDims) != 2 {
		return 0, 0, 0, 0, tensor.ErrShapeMismatch
	}

	batch, outDim, rank, inner := inputDims[0], loraADims[0], loraADims[1], loraBDims[1]
	if loraBDims[0] != rank || inputDims[1] != inner ||
		baseDims[0] != batch || baseDims[1] != outDim ||
		outDims[0] != batch || outDims[1] != outDim {
		return 0, 0, 0, 0, tensor.ErrShapeMismatch
	}

	err := requireProjectionUint32(batch, inner, rank, outDim)
	return batch, inner, rank, outDim, err
}

func requireFusedQKVOutputDims(config metalFusedQKVConfig, batch int, outDim int) error {
	if !config.query.shape.Equal(config.key.shape) || !config.query.shape.Equal(config.value.shape) {
		return tensor.ErrShapeMismatch
	}

	queryDims := config.query.shape.Dims()
	if len(queryDims) != 2 || queryDims[0] != batch || queryDims[1] != outDim {
		return tensor.ErrShapeMismatch
	}

	return nil
}

func requireProjectionUint32(values ...int) error {
	for _, value := range values {
		if value < 0 || int64(value) > math.MaxUint32 {
			return tensor.ErrShapeMismatch
		}
	}

	return nil
}

func requireMetalTensors4(
	first tensor.Tensor,
	second tensor.Tensor,
	third tensor.Tensor,
	fourth tensor.Tensor,
) (*metalTensor, *metalTensor, *metalTensor, *metalTensor, error) {
	tensors, err := requireMetalTensors(first, second, third, fourth)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	return tensors[0], tensors[1], tensors[2], tensors[3], nil
}

func requireMetalTensors(inputs ...tensor.Tensor) ([]*metalTensor, error) {
	tensors := make([]*metalTensor, 0, len(inputs))

	for _, input := range inputs {
		target, err := requireMetalTensor(input)
		if err != nil {
			return nil, err
		}

		tensors = append(tensors, target)
	}

	return tensors, nil
}

func requireMetalProjectionSameDTypeAndBridge(tensors ...*metalTensor) error {
	if len(tensors) == 0 {
		return tensor.ErrShapeMismatch
	}

	storageDType := tensors[0].dtype
	bridge := tensors[0].bridge

	for _, target := range tensors[1:] {
		if target.dtype != storageDType {
			return tensor.ErrDTypeMismatch
		}

		if target.bridge != bridge {
			return errors.New("metal projection: tensors belong to different Metal backends")
		}
	}

	return nil
}
