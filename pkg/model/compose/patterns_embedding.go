package compose

import (
	"strings"

	"github.com/theapemachine/caramba/pkg/manifest"
)

/*
embeddingPattern claims rank-2 tensors that look like embedding
tables. Distinguishing them from regular linear weights is done by
name heuristic — PyTorch and HuggingFace consistently use one of
these suffixes for embeddings:

  wte, wpe                   GPT-2
  embed_tokens, embed_pos    Llama/Qwen/Mistral
  token_embedding            generic
  position_embedding         generic
  shared                     T5

This is universal pattern-matching, not architecture-specific code:
the same heuristic recognizes embeddings in every model that follows
the convention. Adding a new prefix here covers every variant that
adopts it.

The pattern runs at higher priority than linearPattern so it claims
the group before the linear leaf does.
*/
type embeddingPattern struct{}

func (embeddingPattern) Name() string  { return "embedding.token" }
func (embeddingPattern) Priority() int { return 30 }

var embeddingSuffixes = []string{
	"wte", "wpe", "embed_tokens", "embed_pos",
	"token_embedding", "position_embedding", "shared",
}

func (embeddingPattern) Match(group TensorGroup) bool {
	weight, ok := findWithSuffix(group.Names, ".weight")
	if !ok {
		return false
	}

	info, ok := group.Info(weight)
	if !ok || len(info.Shape) != 2 {
		return false
	}

	prefix := group.Prefix
	for _, suffix := range embeddingSuffixes {
		if strings.HasSuffix(prefix, suffix) {
			return true
		}
	}

	return false
}

func (embeddingPattern) Emit(group TensorGroup, ctx *BuilderContext) error {
	weight, _ := findWithSuffix(group.Names, ".weight")
	info, _ := group.Info(weight)
	nodeID := group.Prefix

	node := &manifest.Node{
		ID:   nodeID,
		OpID: "embedding.token",
		Config: map[string]any{
			"vocab_size":            info.Shape[0],
			"d_model":               info.Shape[1],
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
	DefaultRegistry.Register(embeddingPattern{})
}
