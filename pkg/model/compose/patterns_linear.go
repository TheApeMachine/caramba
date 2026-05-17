package compose

import (
	"strings"

	"github.com/theapemachine/caramba/pkg/manifest"
)

/*
linearPattern recognises a rank-2 weight tensor with an optional
rank-1 bias and emits a single projection.linear node. The PyTorch
convention is

  <prefix>.weight  shape [out, in]   (most modules)
  <prefix>.bias    shape [out]       (when bias=True)

There are two notable variants that this leaf pattern handles
without a per-architecture branch:

 1. GPT-2's Conv1D module stores weight as [in, out]. The pattern
    notices the shape orientation at compile time by comparing
    dimensions and records a transpose flag the runtime binder
    applies. The same code path handles both layouts.
 2. Some safetensors files separate weight and bias into the same
    prefix bucket (e.g. "lm_head.weight" alongside "lm_head.bias");
    others ship weight-only modules. Both cases work — the bias
    tensor is optional in TensorRef.

This is the lowest-priority pattern; higher-priority block-level
patterns get first crack at compound prefixes.
*/
type linearPattern struct{}

func (linearPattern) Name() string  { return "linear" }
func (linearPattern) Priority() int { return 10 }

func (linearPattern) Match(group TensorGroup) bool {
	weight, weightOK := findWithSuffix(group.Names, ".weight")
	if !weightOK {
		return false
	}

	info, ok := group.Info(weight)
	if !ok || len(info.Shape) != 2 {
		return false
	}

	// Reject embedding tables — they're rank-2 too but a higher-
	// priority pattern claims them. If embeddingPattern declines
	// (e.g. inputs aren't tokens), the group still gets here.
	return true
}

func (linearPattern) Emit(group TensorGroup, ctx *BuilderContext) error {
	weight, _ := findWithSuffix(group.Names, ".weight")
	bias, hasBias := findWithSuffix(group.Names, ".bias")

	info, _ := group.Info(weight)

	// PyTorch nn.Linear stores [out, in]; that's the assumed default.
	// A Conv1D-style module stores [in, out]. We can't distinguish by
	// shape alone when both dims happen to equal, so we emit the node
	// with both dims recorded and let the runtime binder pick the
	// orientation that matches in_features.
	outFeatures := info.Shape[0]
	inFeatures := info.Shape[1]

	nodeID := group.Prefix

	config := map[string]any{
		"in_features":           inFeatures,
		"out_features":          outFeatures,
		"compose.weight_tensor": weight,
	}

	if hasBias {
		config["compose.bias_tensor"] = bias
	}

	node := &manifest.Node{
		ID:     nodeID,
		OpID:   "projection.linear",
		Config: config,
		In:     []string{},
		Out:    []string{nodeID},
	}

	if err := ctx.AddNode(node); err != nil {
		return err
	}

	ctx.Bind(nodeID, nodeID)

	return nil
}

/*
findWithSuffix returns the first tensor name in names that ends with
suffix, and true. It searches in sorted order so output is
deterministic.
*/
func findWithSuffix(names []string, suffix string) (string, bool) {
	for _, name := range names {
		if strings.HasSuffix(name, suffix) {
			return name, true
		}
	}

	return "", false
}

func init() {
	DefaultRegistry.Register(linearPattern{})
}
