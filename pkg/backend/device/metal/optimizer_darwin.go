//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "bridge_darwin.h"
*/
import "C"

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

type metalOptimizer4Config struct {
	params       *metalTensor
	gradients    *metalTensor
	firstState   *metalTensor
	secondState  *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	count        uint32
}

type metalOptimizer3Config struct {
	params       *metalTensor
	gradients    *metalTensor
	state        *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	count        uint32
}

type metalOptimizer2Config struct {
	params       *metalTensor
	gradients    *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	count        uint32
}

func runMetalOptimizer4Kernel(operation metalOptimizerOp, args ...tensor.Tensor) error {
	if len(args) != 5 {
		return tensor.ErrShapeMismatch
	}

	return runMetalOptimizer4(operation, args[0], args[1], args[2], args[3], args[4])
}

func runMetalOptimizer3Kernel(operation metalOptimizerOp, args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalOptimizer3(operation, args[0], args[1], args[2], args[3])
}

func runMetalOptimizer2Kernel(operation metalOptimizerOp, args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalOptimizer2(operation, args[0], args[1], args[2])
}

func runMetalOptimizer4(
	operation metalOptimizerOp,
	params tensor.Tensor,
	gradients tensor.Tensor,
	firstState tensor.Tensor,
	secondState tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalOptimizer4(params, gradients, firstState, secondState, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(
		config.out, config.params, config.gradients, config.firstState, config.secondState,
	)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_optimizer4(
		config.params.bridge.device,
		C.int(operation),
		C.int(config.elementDType),
		config.params.buffer,
		config.gradients.buffer,
		config.firstState.buffer,
		config.secondState.buffer,
		config.out.buffer,
		C.uint32_t(config.count),
		C.uint64_t(token),
		&status,
	)

	return finishMetalOptimizerDispatch("optimizer4", token, rc, status)
}

func runMetalOptimizer3(
	operation metalOptimizerOp,
	params tensor.Tensor,
	gradients tensor.Tensor,
	state tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalOptimizer3(params, gradients, state, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.params, config.gradients, config.state)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_optimizer3(
		config.params.bridge.device,
		C.int(operation),
		C.int(config.elementDType),
		config.params.buffer,
		config.gradients.buffer,
		config.state.buffer,
		config.out.buffer,
		C.uint32_t(config.count),
		C.uint64_t(token),
		&status,
	)

	return finishMetalOptimizerDispatch("optimizer3", token, rc, status)
}

func runMetalOptimizer2(
	operation metalOptimizerOp,
	params tensor.Tensor,
	gradients tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalOptimizer2(params, gradients, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.params, config.gradients)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_optimizer2(
		config.params.bridge.device,
		C.int(operation),
		C.int(config.elementDType),
		config.params.buffer,
		config.gradients.buffer,
		config.out.buffer,
		C.uint32_t(config.count),
		C.uint64_t(token),
		&status,
	)

	return finishMetalOptimizerDispatch("optimizer2", token, rc, status)
}

func finishMetalOptimizerDispatch(name string, token uint64, rc C.int, status C.MetalStatus) error {
	if rc == 0 {
		return nil
	}

	err := fmt.Errorf("metal %s: %s", name, metalStatus("dispatch", status))
	metalCompletions.Fail(token, err)
	return err
}
