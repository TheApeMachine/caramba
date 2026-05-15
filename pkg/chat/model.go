package chat

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
ModelConfig configures the future CausalLM chat generator.
*/
type ModelConfig struct {
	Model           string
	TokenizerSource tokenizer.Source
	MaxNewTokens    int
}

/*
ModelGenerator is the slot where the CausalLM runtime will stream logits.
*/
type ModelGenerator struct {
	config ModelConfig
}

/*
NewModelGenerator validates model chat configuration.
*/
func NewModelGenerator(_ context.Context, config ModelConfig) (*ModelGenerator, error) {
	if strings.TrimSpace(config.Model) == "" {
		return nil, fmt.Errorf("chat.model: model is required")
	}

	if config.MaxNewTokens < 1 {
		return nil, fmt.Errorf("chat.model: max_new_tokens must be positive")
	}

	return nil, fmt.Errorf(
		"chat.model: CausalLM runtime is not connected yet for %q",
		config.Model,
	)
}

/*
Generate streams model tokens.
*/
func (generator *ModelGenerator) Generate(
	context.Context,
	string,
	func(string) error,
) error {
	return fmt.Errorf(
		"chat.model: CausalLM runtime is not connected yet for %q",
		generator.config.Model,
	)
}
