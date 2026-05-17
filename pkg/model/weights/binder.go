package weights

import (
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
)

type Binder struct {
	store *Store
}

func NewBinder(store *Store) *Binder {
	return &Binder{store: store}
}

func BindIR(graph *ir.Graph, store *Store) error {
	return NewBinder(store).BindIR(graph)
}

func (binder *Binder) BindIR(graph *ir.Graph) error {
	if graph == nil {
		return fmt.Errorf("weights: graph is required")
	}

	if binder == nil || binder.store == nil {
		return fmt.Errorf("weights: store is required")
	}

	for _, node := range graph.Nodes() {
		if err := binder.bindNode(node); err != nil {
			return err
		}
	}

	return nil
}

func (binder *Binder) bindNode(node *ir.Node) error {
	if handled, err := binder.bindFromMetadataKeys(node); handled || err != nil {
		return err
	}

	switch node.OperationID() {
	case ir.OpEmbeddingToken:
		return binder.bindVectorOrMatrix(
			node,
			"weight",
			exactTensorNames(node.ID(), "weight"),
		)
	case ir.OpMathLayerNorm:
		if err := binder.bindVectorOrMatrix(
			node,
			"weight",
			exactTensorNames(node.ID(), "weight"),
		); err != nil {
			return err
		}

		return binder.bindVectorOrMatrix(
			node,
			"bias",
			exactTensorNames(node.ID(), "bias"),
		)
	case ir.OpMathRMSNorm:
		if !nodeRMSNormAffine(node) {
			return nil
		}

		return binder.bindVectorOrMatrix(
			node,
			"weight",
			exactTensorNames(node.ID(), "weight"),
		)
	case ir.OpMathGroupNorm:
		if err := binder.bindVectorOrMatrix(
			node,
			"weight",
			exactTensorNames(node.ID(), "weight"),
		); err != nil {
			return err
		}

		return binder.bindVectorOrMatrix(
			node,
			"bias",
			exactTensorNames(node.ID(), "bias"),
		)
	case ir.OpProjectionLinear:
		return binder.bindLinear(node)
	case ir.OpConvolutionConv2D:
		return binder.bindConv2D(node)
	case ir.OpConvolutionConvT2D:
		return binder.bindConvTranspose2D(node)
	default:
		return nil
	}
}

func (binder *Binder) bindVectorOrMatrix(
	node *ir.Node, metadataKey string, names []string,
) error {
	name, ok := binder.first(names)

	if !ok {
		return fmt.Errorf("weights: no tensor found for node %q %s", node.ID(), metadataKey)
	}

	values, err := binder.store.Values(name)

	if err != nil {
		return err
	}

	node.SetMetadata(metadataKey, values)

	return nil
}

func (binder *Binder) bindLinear(node *ir.Node) error {
	inFeatures := nodeConfigInt(node, "in_features")
	outFeatures := nodeConfigInt(node, "out_features")

	if inFeatures <= 0 || outFeatures <= 0 {
		return fmt.Errorf("weights: node %q linear dimensions are required", node.ID())
	}

	weightName, ok := binder.first(exactTensorNames(node.ID(), "weight"))

	if !ok {
		return fmt.Errorf("weights: no tensor found for node %q weight", node.ID())
	}

	weight, err := binder.linearWeight(weightName, inFeatures, outFeatures)

	if err != nil {
		return fmt.Errorf("weights: node %q: %w", node.ID(), err)
	}

	node.SetMetadata("weight", weight)

	if biasName, ok := binder.first(exactTensorNames(node.ID(), "bias")); ok {
		bias, err := binder.linearBias(biasName, outFeatures)

		if err != nil {
			return fmt.Errorf("weights: node %q: %w", node.ID(), err)
		}

		node.SetMetadata("bias", bias)
	}

	return nil
}

func (binder *Binder) linearWeight(name string, inFeatures, outFeatures int) ([]float64, error) {
	key := fmt.Sprintf("linear:%s:%d:%d", name, inFeatures, outFeatures)

	return binder.store.Derived(key, func() ([]float64, error) {
		info, ok := binder.store.Info(name)

		if !ok {
			return nil, fmt.Errorf("tensor %q not found", name)
		}

		values, err := binder.store.Values(name)

		if err != nil {
			return nil, err
		}

		return orientLinearWeight(name, values, info.Shape, inFeatures, outFeatures)
	})
}

func (binder *Binder) linearBias(name string, outFeatures int) ([]float64, error) {
	values, err := binder.store.Values(name)

	if err != nil {
		return nil, err
	}

	if len(values) != outFeatures {
		return nil, fmt.Errorf(
			"bias %q length %d does not match out_features %d",
			name, len(values), outFeatures,
		)
	}

	return values, nil
}

func (binder *Binder) bindConv2D(node *ir.Node) error {
	inChannels := nodeConfigIntAny(node, "in_channels", "in_c")
	outChannels := nodeConfigIntAny(node, "out_channels", "out_c")
	kernelH := nodeConfigIntAny(node, "kernel_h", "k_h")
	kernelW := nodeConfigIntAny(node, "kernel_w", "k_w")
	groups := nodeConfigIntAny(node, "groups")

	if groups == 0 {
		groups = 1
	}

	expectedWeight := 0

	if inChannels > 0 && outChannels > 0 && kernelH > 0 && kernelW > 0 && groups > 0 {
		expectedWeight = outChannels * (inChannels / groups) * kernelH * kernelW
	}

	if err := binder.bindTensor(
		node,
		"weight",
		exactTensorNames(node.ID(), "weight"),
		expectedWeight,
	); err != nil {
		return err
	}

	return binder.bindTensor(
		node,
		"bias",
		exactTensorNames(node.ID(), "bias"),
		outChannels,
	)
}

func (binder *Binder) bindConvTranspose2D(node *ir.Node) error {
	inChannels := nodeConfigIntAny(node, "in_channels", "in_c")
	outChannels := nodeConfigIntAny(node, "out_channels", "out_c")
	kernelH := nodeConfigIntAny(node, "kernel_h", "k_h")
	kernelW := nodeConfigIntAny(node, "kernel_w", "k_w")
	groups := nodeConfigIntAny(node, "groups")

	if groups == 0 {
		groups = 1
	}

	expectedWeight := 0

	if inChannels > 0 && outChannels > 0 && kernelH > 0 && kernelW > 0 && groups > 0 {
		expectedWeight = inChannels * (outChannels / groups) * kernelH * kernelW
	}

	if err := binder.bindTensor(
		node,
		"weight",
		exactTensorNames(node.ID(), "weight"),
		expectedWeight,
	); err != nil {
		return err
	}

	return binder.bindTensor(
		node,
		"bias",
		exactTensorNames(node.ID(), "bias"),
		outChannels,
	)
}

func (binder *Binder) bindTensor(
	node *ir.Node,
	metadataKey string,
	names []string,
	expectedLength int,
) error {
	name, ok := binder.first(names)

	if !ok {
		return fmt.Errorf("weights: no tensor found for node %q %s", node.ID(), metadataKey)
	}

	values, err := binder.store.Values(name)

	if err != nil {
		return err
	}

	if expectedLength > 0 && len(values) != expectedLength {
		return fmt.Errorf(
			"weights: node %q %s length %d does not match expected %d",
			node.ID(),
			metadataKey,
			len(values),
			expectedLength,
		)
	}

	node.SetMetadata(metadataKey, values)

	return nil
}

func (binder *Binder) first(names []string) (string, bool) {
	for _, name := range names {
		if binder.store.Has(name) {
			return name, true
		}
	}

	return "", false
}

func orientLinearWeight(
	name string,
	values []float64,
	shape []int,
	inFeatures,
	outFeatures int,
) ([]float64, error) {
	if len(values) != inFeatures*outFeatures {
		return nil, fmt.Errorf(
			"weight length %d does not match in_features*out_features=%d",
			len(values), inFeatures*outFeatures,
		)
	}

	if len(shape) != 2 {
		return values, nil
	}

	if torchLinearWeight(name) {
		if shape[0] != outFeatures || shape[1] != inFeatures {
			return nil, fmt.Errorf(
				"weight shape %v does not match torch linear [%d %d]",
				shape,
				outFeatures,
				inFeatures,
			)
		}

		return transpose(values, outFeatures, inFeatures), nil
	}

	switch {
	case shape[0] == inFeatures && shape[1] == outFeatures:
		return values, nil
	case shape[0] == outFeatures && shape[1] == inFeatures:
		return transpose(values, outFeatures, inFeatures), nil
	default:
		return nil, fmt.Errorf(
			"weight shape %v does not match [%d %d] or [%d %d]",
			shape, inFeatures, outFeatures, outFeatures, inFeatures,
		)
	}
}

func torchLinearWeight(name string) bool {
	if name == "lm_head.weight" || name == "model.embed_tokens.weight" {
		return true
	}

	return strings.HasPrefix(name, "model.layers.") ||
		strings.HasPrefix(name, "transformer_blocks.") ||
		strings.HasPrefix(name, "single_transformer_blocks.") ||
		strings.HasPrefix(name, "context_embedder.") ||
		strings.HasPrefix(name, "x_embedder.") ||
		strings.HasPrefix(name, "norm_out.linear.") ||
		strings.HasPrefix(name, "proj_out.")
}

func linearInputFeatures(shape []int, outFeatures int) (int, error) {
	if len(shape) != 2 {
		return 0, fmt.Errorf("linear tensor must be rank 2, got %v", shape)
	}

	if shape[0] == outFeatures {
		return shape[1], nil
	}

	if shape[1] == outFeatures {
		return shape[0], nil
	}

	return 0, fmt.Errorf("linear tensor shape %v does not include out_features %d", shape, outFeatures)
}

func slicePackedLinearRows(
	values []float64,
	shape []int,
	inFeatures int,
	outFeatures int,
	start int,
	width int,
) ([]float64, error) {
	if len(shape) != 2 {
		return nil, fmt.Errorf("packed tensor must be rank 2, got %v", shape)
	}

	switch {
	case shape[0] == inFeatures && shape[1] >= start+width:
		return sliceColumns(values, inFeatures, shape[1], start, width), nil
	case shape[1] == inFeatures && shape[0] >= start+width:
		rows := sliceRows(values, shape[0], inFeatures, start, width)

		return transpose(rows, width, inFeatures), nil
	default:
		return nil, fmt.Errorf(
			"packed tensor shape %v cannot provide [%d %d] slice at %d",
			shape,
			inFeatures,
			outFeatures,
			start,
		)
	}
}

func sliceColumns(values []float64, rows, cols, start, width int) []float64 {
	out := make([]float64, rows*width)

	for row := 0; row < rows; row++ {
		copy(out[row*width:(row+1)*width], values[row*cols+start:row*cols+start+width])
	}

	return out
}

func sliceRows(values []float64, rows, cols, start, height int) []float64 {
	out := make([]float64, height*cols)

	for row := 0; row < height; row++ {
		copy(out[row*cols:(row+1)*cols], values[(start+row)*cols:(start+row+1)*cols])
	}

	return out
}

func transpose(values []float64, rows, cols int) []float64 {
	out := make([]float64, len(values))

	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			out[col*rows+row] = values[row*cols+col]
		}
	}

	return out
}

func nodeConfigInt(node *ir.Node, key string) int {
	value := node.Metadata()[key]

	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	case float32:
		return int(typed)
	default:
		return 0
	}
}

func nodeConfigIntAny(node *ir.Node, keys ...string) int {
	for _, key := range keys {
		if value := nodeConfigInt(node, key); value != 0 {
			return value
		}
	}

	return 0
}

func nodeRMSNormAffine(node *ir.Node) bool {
	affine := nodeConfigBool(node, "affine", true)

	return nodeConfigBool(node, "elementwise_affine", affine)
}

func nodeConfigBool(node *ir.Node, key string, fallback bool) bool {
	value := node.Metadata()[key]

	typed, ok := value.(bool)

	if !ok {
		return fallback
	}

	return typed
}
