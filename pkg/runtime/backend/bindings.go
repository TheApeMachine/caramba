package backend

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/state"
)

/*
deriveDefaultShape returns the shape used when lowering the manifest
graph. The convention is to use the input named "input_ids" (a token
sequence) when present: the shape becomes [1, len(tokens)] so the
manifest's token-driven graph constructs match real prefill/decode
behavior. If no input_ids is supplied the GraphModule.Config may
provide a `default_shape` []int; otherwise a [1] scalar shape is
used so manifests with no sequence dependence still lower cleanly.
*/
func deriveDefaultShape(
	module program.GraphModule, inputs map[string]any,
) (tensor.Shape, error) {
	if value, ok := inputs["input_ids"]; ok {
		tokens, err := asIntSlice(value)

		if err == nil && len(tokens) > 0 {
			return tensor.NewShape([]int{1, len(tokens)})
		}

		switch value.(type) {
		case int, int32, int64:
			return tensor.NewShape([]int{1, 1})
		}
	}

	if raw, ok := module.Config["default_shape"]; ok {
		dims, err := asIntSlice(raw)

		if err != nil {
			return tensor.Shape{}, fmt.Errorf("default_shape: %w", err)
		}

		if len(dims) > 0 {
			return tensor.NewShape(dims)
		}
	}

	if length, ok := singleInputLength(inputs); ok {
		return tensor.NewShape([]int{length})
	}

	return tensor.NewShape([]int{1})
}

/*
deriveInputShapes turns the GraphModule.Config["input_shapes"] entry
into a per-input map of tensor.Shape values. The format is:

  input_shapes: { encoder_hidden_states: [1, 256, 7680], timestep: [1] }

This is the runtime equivalent of LowerGraphToIRWithInputShapes,
which lets multi-input graphs (diffusion denoisers, multi-encoder
conditioning) declare per-input shapes per call.
*/
func deriveInputShapes(
	module program.GraphModule, inputs map[string]any,
) (map[string]tensor.Shape, error) {
	raw, ok := module.Config["input_shapes"]

	if !ok {
		return nil, nil
	}

	entries, err := shapeMap(raw)

	if err != nil {
		return nil, fmt.Errorf("input_shapes: %w", err)
	}

	out := make(map[string]tensor.Shape, len(entries))

	for name, dims := range entries {
		resolved, err := resolveDynamicDims(dims, inputs[name])

		if err != nil {
			return nil, fmt.Errorf("input_shapes.%s: %w", name, err)
		}

		shape, err := tensor.NewShape(resolved)

		if err != nil {
			return nil, fmt.Errorf("input_shapes.%s: %w", name, err)
		}

		out[name] = shape
	}

	return out, nil
}

/*
resolveDynamicDims replaces a single -1 entry in dims with the size
that makes the shape consume exactly len(value) elements. This lets
manifests express "this dimension is the natural data dimension" —
typically the text-sequence axis on diffusion encoder_hidden_states,
where the actual length depends on the prompt and cannot be known at
manifest authoring time. At most one dim may be -1; the rest are
treated as fixed.
*/
func resolveDynamicDims(dims []int, value any) ([]int, error) {
	dynamicIndex := -1
	knownProduct := 1

	for index, dim := range dims {
		if dim == -1 {
			if dynamicIndex >= 0 {
				return nil, fmt.Errorf("at most one dimension may be -1")
			}

			dynamicIndex = index

			continue
		}

		if dim <= 0 {
			return nil, fmt.Errorf("dimension %d must be positive, got %d", index, dim)
		}

		knownProduct *= dim
	}

	if dynamicIndex < 0 {
		return dims, nil
	}

	if value == nil {
		return nil, fmt.Errorf("dimension -1 requires the input value to derive size, got nil")
	}

	values, err := coerceValues(value)

	if err != nil {
		return nil, err
	}

	if values == nil {
		return nil, fmt.Errorf("dimension -1 requires a non-nil input value")
	}

	if knownProduct <= 0 || len(values)%knownProduct != 0 {
		return nil, fmt.Errorf(
			"input length %d is not divisible by the product of fixed dimensions (%d)",
			len(values), knownProduct,
		)
	}

	resolved := make([]int, len(dims))
	copy(resolved, dims)
	resolved[dynamicIndex] = len(values) / knownProduct

	return resolved, nil
}

func shapeMap(raw any) (map[string][]int, error) {
	switch typed := raw.(type) {
	case map[string][]int:
		out := map[string][]int{}

		for key, dims := range typed {
			out[key] = append([]int(nil), dims...)
		}

		return out, nil
	case map[string]any:
		out := map[string][]int{}

		for key, value := range typed {
			dims, err := asIntSlice(value)

			if err != nil {
				return nil, fmt.Errorf("%s: %w", key, err)
			}

			out[key] = dims
		}

		return out, nil
	}

	return nil, fmt.Errorf("expected map[string][]int, got %T", raw)
}

/*
singleInputLength returns the element count of the lone non-"graph"
input when exactly one such input is supplied. This lets manifests
with a single 1-D external input lower correctly without forcing the
program author to declare default_shape.
*/
func singleInputLength(inputs map[string]any) (int, bool) {
	if len(inputs) == 0 {
		return 0, false
	}

	candidates := map[string]any{}

	for name, value := range inputs {
		if name == "graph" {
			continue
		}

		candidates[name] = value
	}

	if len(candidates) != 1 {
		return 0, false
	}

	for _, value := range candidates {
		values, err := coerceValues(value)

		if err != nil || values == nil {
			return 0, false
		}

		return len(values), true
	}

	return 0, false
}

/*
bindAllInputs walks the runtime-supplied inputs and binds each onto
the IR input node whose ID matches the input name. Values may be
[]float64, []int, or runtime state objects; the helpers below
normalize them onto float64 vectors the IR consumes.

The "graph" input is excluded because it identifies the graph itself
rather than a tensor.
*/
func bindAllInputs(index *ir.Index, inputs map[string]any) error {
	for name, value := range inputs {
		if name == "graph" {
			continue
		}

		if err := bindOne(index, name, value); err != nil {
			return fmt.Errorf("input %q: %w", name, err)
		}
	}

	return nil
}

func bindOne(index *ir.Index, name string, value any) error {
	node := index.Node(name)

	if node == nil {
		return nil
	}

	values, err := coerceValues(value)

	if err != nil {
		return err
	}

	if values == nil {
		return nil
	}

	if node.OpType() != ir.OpInput {
		return fmt.Errorf("IR node %q is not an input", name)
	}

	node.SetMetadata("values", values)

	return nil
}

func coerceValues(value any) ([]float64, error) {
	switch typed := value.(type) {
	case []float64:
		return append([]float64(nil), typed...), nil
	case []float32:
		out := make([]float64, len(typed))

		for index, element := range typed {
			out[index] = float64(element)
		}

		return out, nil
	case []int:
		out := make([]float64, len(typed))

		for index, element := range typed {
			out[index] = float64(element)
		}

		return out, nil
	case int:
		return []float64{float64(typed)}, nil
	case int64:
		return []float64{float64(typed)}, nil
	case float64:
		return []float64{typed}, nil
	case *state.Tensor:
		return typed.Values(), nil
	case *state.TokenBuffer:
		tokens := typed.Tokens()
		out := make([]float64, len(tokens))

		for index, token := range tokens {
			out[index] = float64(token)
		}

		return out, nil
	case *state.Counter:
		return []float64{float64(typed.Value())}, nil
	case nil:
		return nil, nil
	}

	return nil, fmt.Errorf("unsupported input type %T", value)
}

func asIntSlice(value any) ([]int, error) {
	switch typed := value.(type) {
	case []int:
		return append([]int(nil), typed...), nil
	case []any:
		out := make([]int, len(typed))

		for index, element := range typed {
			switch elementTyped := element.(type) {
			case int:
				out[index] = elementTyped
			case int64:
				out[index] = int(elementTyped)
			case float64:
				out[index] = int(elementTyped)
			default:
				return nil, fmt.Errorf("element %d: expected integer, got %T", index, element)
			}
		}

		return out, nil
	}

	return nil, fmt.Errorf("expected []int, got %T", value)
}
