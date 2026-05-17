package weights

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
)

type tensorBinding struct {
	weightName string
	biasName   string
	sliceAxis  string
	sliceStart int
	sliceSize  int
}

func (binder *Binder) bindFromMetadataKeys(node *ir.Node) (bool, error) {
	binding, handled, err := tensorBindingFromMetadata(node.Metadata())

	if err != nil || !handled {
		return handled, err
	}

	if binding.weightName == "" {
		return true, fmt.Errorf(
			"weights: from_safetensors weight tensor is required for node %q",
			node.ID(),
		)
	}

	if !binder.store.Has(binding.weightName) {
		return true, fmt.Errorf(
			"weights: tensor %q not found for node %q",
			binding.weightName,
			node.ID(),
		)
	}

	if node.OperationID() == ir.OpProjectionLinear {
		return true, binder.bindLinearFromBinding(node, binding)
	}

	values, err := binder.store.Values(binding.weightName)

	if err != nil {
		return true, err
	}

	node.SetMetadata("weight", values)

	if binding.biasName == "" {
		return true, nil
	}

	bias, err := binder.store.Values(binding.biasName)

	if err != nil {
		return true, err
	}

	node.SetMetadata("bias", bias)

	return true, nil
}

func (binder *Binder) bindLinearFromBinding(
	node *ir.Node,
	binding tensorBinding,
) error {
	inFeatures := nodeConfigInt(node, "in_features")
	outFeatures := nodeConfigInt(node, "out_features")

	if inFeatures <= 0 || outFeatures <= 0 {
		return fmt.Errorf("weights: node %q linear dimensions are required", node.ID())
	}

	info, _ := binder.store.Info(binding.weightName)
	values, err := binder.store.Values(binding.weightName)

	if err != nil {
		return err
	}

	weight, err := binder.store.Derived(
		fmt.Sprintf(
			"direct-linear:%s:%s:%d:%d:%d:%d",
			binding.weightName,
			binding.sliceAxis,
			inFeatures,
			outFeatures,
			binding.sliceStart,
			binding.sliceSize,
		),
		func() ([]float64, error) {
			return linearBindingWeight(
				binding,
				values,
				info.Shape,
				inFeatures,
				outFeatures,
			)
		},
	)

	if err != nil {
		return fmt.Errorf("weights: node %q: %w", node.ID(), err)
	}

	node.SetMetadata("weight", weight)

	if binding.biasName == "" {
		return nil
	}

	biasValues, err := binder.store.Values(binding.biasName)

	if err != nil {
		return err
	}

	bias, err := linearBindingBias(binding, biasValues, outFeatures)

	if err != nil {
		return fmt.Errorf("weights: node %q: %w", node.ID(), err)
	}

	node.SetMetadata("bias", bias)

	return nil
}

func linearBindingWeight(
	binding tensorBinding,
	values []float64,
	shape []int,
	inFeatures int,
	outFeatures int,
) ([]float64, error) {
	axis := strings.TrimSpace(binding.sliceAxis)

	if axis == "" {
		return orientLinearWeight(
			binding.weightName,
			values,
			shape,
			inFeatures,
			outFeatures,
		)
	}

	if binding.sliceStart < 0 {
		return nil, fmt.Errorf("weight slice start must be non-negative")
	}

	switch axis {
	case "output":
		size := binding.sliceSize

		if size == 0 {
			size = outFeatures
		}

		if size != outFeatures {
			return nil, fmt.Errorf(
				"output slice size %d does not match out_features %d",
				size,
				outFeatures,
			)
		}

		return slicePackedLinearRows(
			values,
			shape,
			inFeatures,
			outFeatures,
			binding.sliceStart,
			size,
		)
	case "input":
		return linearInputSliceWeight(binding, values, shape, inFeatures, outFeatures)
	default:
		return nil, fmt.Errorf("unsupported weight slice axis %q", axis)
	}
}

func linearInputSliceWeight(
	binding tensorBinding,
	values []float64,
	shape []int,
	inFeatures int,
	outFeatures int,
) ([]float64, error) {
	size := binding.sliceSize

	if size == 0 {
		size = inFeatures
	}

	if size != inFeatures {
		return nil, fmt.Errorf(
			"input slice size %d does not match in_features %d",
			size,
			inFeatures,
		)
	}

	totalInFeatures, err := linearInputFeatures(shape, outFeatures)

	if err != nil {
		return nil, err
	}

	if binding.sliceStart+size > totalInFeatures {
		return nil, fmt.Errorf(
			"input slice [%d:%d] exceeds input width %d",
			binding.sliceStart,
			binding.sliceStart+size,
			totalInFeatures,
		)
	}

	weight, err := orientLinearWeight(
		binding.weightName,
		values,
		shape,
		totalInFeatures,
		outFeatures,
	)

	if err != nil {
		return nil, err
	}

	return sliceRows(
		weight,
		totalInFeatures,
		outFeatures,
		binding.sliceStart,
		size,
	), nil
}

func linearBindingBias(
	binding tensorBinding,
	values []float64,
	outFeatures int,
) ([]float64, error) {
	if strings.TrimSpace(binding.sliceAxis) != "output" {
		if len(values) != outFeatures {
			return nil, fmt.Errorf(
				"bias %q length %d does not match out_features %d",
				binding.biasName,
				len(values),
				outFeatures,
			)
		}

		return values, nil
	}

	if binding.sliceStart < 0 {
		return nil, fmt.Errorf("bias slice start must be non-negative")
	}

	size := binding.sliceSize

	if size == 0 {
		size = outFeatures
	}

	if size != outFeatures {
		return nil, fmt.Errorf(
			"bias output slice size %d does not match out_features %d",
			size,
			outFeatures,
		)
	}

	if binding.sliceStart+size > len(values) {
		return nil, fmt.Errorf(
			"bias output slice [%d:%d] exceeds length %d",
			binding.sliceStart,
			binding.sliceStart+size,
			len(values),
		)
	}

	return append(
		[]float64(nil),
		values[binding.sliceStart:binding.sliceStart+size]...,
	), nil
}

func tensorBindingFromMetadata(
	metadata map[string]any,
) (tensorBinding, bool, error) {
	if binding, ok := composeTensorBinding(metadata); ok {
		return binding, true, nil
	}

	raw, ok := metadata["from_safetensors"]

	if ok {
		mapping, ok := raw.(map[string]any)

		if !ok {
			return tensorBinding{}, true, fmt.Errorf(
				"from_safetensors metadata must be a mapping, got %T",
				raw,
			)
		}

		binding, err := mappedTensorBinding(mapping)

		return binding, true, err
	}

	binding, ok, err := prefixedTensorBinding(metadata)

	return binding, ok, err
}

func composeTensorBinding(metadata map[string]any) (tensorBinding, bool) {
	weightName, _ := metadata["compose.weight_tensor"].(string)

	if strings.TrimSpace(weightName) == "" {
		return tensorBinding{}, false
	}

	biasName, _ := metadata["compose.bias_tensor"].(string)

	return tensorBinding{
		weightName: strings.TrimSpace(weightName),
		biasName:   strings.TrimSpace(biasName),
	}, true
}

func prefixedTensorBinding(
	metadata map[string]any,
) (tensorBinding, bool, error) {
	weightName := metadataString(
		metadata,
		"safetensors.weight",
		"safetensors.weight_tensor",
	)

	if weightName == "" {
		return tensorBinding{}, false, nil
	}

	binding := tensorBinding{
		weightName: weightName,
		biasName: metadataString(
			metadata,
			"safetensors.bias",
			"safetensors.bias_tensor",
		),
		sliceAxis: metadataString(
			metadata,
			"safetensors.slice_axis",
			"safetensors.axis",
		),
	}

	var err error
	binding.sliceStart, err = metadataInt(
		metadata,
		"safetensors.slice_start",
		"safetensors.start",
	)

	if err != nil {
		return tensorBinding{}, true, err
	}

	binding.sliceSize, err = metadataInt(
		metadata,
		"safetensors.slice_size",
		"safetensors.size",
	)

	return binding, true, err
}

func mappedTensorBinding(mapping map[string]any) (tensorBinding, error) {
	binding := tensorBinding{
		weightName: metadataString(mapping, "weight", "weight_tensor"),
		biasName:   metadataString(mapping, "bias", "bias_tensor"),
		sliceAxis:  metadataString(mapping, "slice_axis", "axis"),
	}

	var err error
	binding.sliceStart, err = metadataInt(mapping, "slice_start", "start")

	if err != nil {
		return tensorBinding{}, err
	}

	binding.sliceSize, err = metadataInt(mapping, "slice_size", "size")

	return binding, err
}

func metadataString(metadata map[string]any, keys ...string) string {
	for _, key := range keys {
		value, ok := metadata[key]

		if !ok {
			continue
		}

		text, ok := value.(string)

		if ok {
			return strings.TrimSpace(text)
		}
	}

	return ""
}

func metadataInt(metadata map[string]any, keys ...string) (int, error) {
	for _, key := range keys {
		value, ok := metadata[key]

		if !ok || value == nil {
			continue
		}

		switch typed := value.(type) {
		case int:
			return typed, nil
		case int64:
			return int(typed), nil
		case float64:
			if typed != float64(int(typed)) {
				return 0, fmt.Errorf("%s must be an integer, got %v", key, typed)
			}

			return int(typed), nil
		case string:
			trimmed := strings.TrimSpace(typed)

			if trimmed == "" {
				return 0, nil
			}

			parsed, err := strconv.Atoi(trimmed)

			if err != nil {
				return 0, fmt.Errorf("%s must be an integer, got %q", key, typed)
			}

			return parsed, nil
		default:
			return 0, fmt.Errorf("%s must be an integer, got %T", key, value)
		}
	}

	return 0, nil
}

func exactTensorNames(nodeID string, suffix string) []string {
	return []string{nodeID + "." + suffix}
}
