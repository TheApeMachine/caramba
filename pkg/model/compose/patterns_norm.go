package compose

import (
	"github.com/theapemachine/caramba/pkg/manifest"
)

const (
	defaultLayerNormEpsilon = 1e-5
	defaultRMSNormEpsilon   = 1e-6
)

/*
layerNormPattern matches a rank-1 weight + rank-1 bias pair and
emits a math.layernorm node. PyTorch's LayerNorm stores both
tensors at the module's prefix:

	<prefix>.weight  shape [dim]
	<prefix>.bias    shape [dim]

If only weight is present (no bias), rmsNormPattern picks the group
up instead.
*/
type layerNormPattern struct{}

func (layerNormPattern) Name() string  { return "layernorm" }
func (layerNormPattern) Priority() int { return 20 }

func (layerNormPattern) Match(group TensorGroup) bool {
	weight, weightOK := findWithSuffix(group.Names, ".weight")
	bias, biasOK := findWithSuffix(group.Names, ".bias")

	if !weightOK || !biasOK {
		return false
	}

	wInfo, ok := group.Info(weight)
	if !ok || len(wInfo.Shape) != 1 {
		return false
	}

	bInfo, ok := group.Info(bias)
	if !ok || len(bInfo.Shape) != 1 || bInfo.Shape[0] != wInfo.Shape[0] {
		return false
	}

	return true
}

func (layerNormPattern) Emit(group TensorGroup, ctx *BuilderContext) error {
	weight, _ := findWithSuffix(group.Names, ".weight")
	bias, _ := findWithSuffix(group.Names, ".bias")

	wInfo, _ := group.Info(weight)
	nodeID := group.Prefix

	epsilon, err := ctx.hints.GetFloat(HintLayerNormEpsilon, defaultLayerNormEpsilon)

	if err != nil {
		return err
	}

	node := &manifest.Node{
		ID:   nodeID,
		OpID: "math.layernorm",
		Config: map[string]any{
			"normalized_shape":      []int{wInfo.Shape[0]},
			"eps":                   epsilon,
			"compose.weight_tensor": weight,
			"compose.bias_tensor":   bias,
		},
		In:  []string{},
		Out: []string{nodeID},
	}

	if err := ctx.AddNode(node); err != nil {
		return err
	}

	ctx.Bind(nodeID, nodeID)

	return nil
}

/*
rmsNormPattern matches a lone rank-1 weight (no bias) and emits a
math.rmsnorm node. Modern transformers (Llama, Qwen, FLUX) use this
in place of LayerNorm.
*/
type rmsNormPattern struct{}

func (rmsNormPattern) Name() string  { return "rmsnorm" }
func (rmsNormPattern) Priority() int { return 15 }

func (rmsNormPattern) Match(group TensorGroup) bool {
	if _, hasBias := findWithSuffix(group.Names, ".bias"); hasBias {
		return false
	}

	weight, ok := findWithSuffix(group.Names, ".weight")
	if !ok {
		return false
	}

	wInfo, ok := group.Info(weight)
	if !ok || len(wInfo.Shape) != 1 {
		return false
	}

	return true
}

func (rmsNormPattern) Emit(group TensorGroup, ctx *BuilderContext) error {
	weight, _ := findWithSuffix(group.Names, ".weight")
	wInfo, _ := group.Info(weight)
	nodeID := group.Prefix

	epsilon, err := ctx.hints.GetFloat(HintRMSNormEpsilon, defaultRMSNormEpsilon)

	if err != nil {
		return err
	}

	node := &manifest.Node{
		ID:   nodeID,
		OpID: "math.rmsnorm",
		Config: map[string]any{
			"normalized_shape":      []int{wInfo.Shape[0]},
			"eps":                   epsilon,
			"affine":                true,
			"compose.weight_tensor": weight,
		},
		In:  []string{},
		Out: []string{nodeID},
	}

	if err := ctx.AddNode(node); err != nil {
		return err
	}

	ctx.Bind(nodeID, nodeID)

	return nil
}

func init() {
	DefaultRegistry.Register(layerNormPattern{})
	DefaultRegistry.Register(rmsNormPattern{})
}
