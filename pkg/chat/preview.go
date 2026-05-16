package chat

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
PreviewConfig configures the tokenizer-backed preview generator.
*/
type PreviewConfig struct {
	TokenizerSource tokenizer.Source
	StreamDelay     time.Duration
}

/*
PreviewGenerator streams an honest tokenizer-backed response before manifest
execution is connected to model weights.
*/
type PreviewGenerator struct {
	artifact    *tokenizer.Artifact
	streamDelay time.Duration
}

/*
NewPreviewGenerator instantiates a preview generator.
*/
func NewPreviewGenerator(ctx context.Context, config PreviewConfig) (*PreviewGenerator, error) {
	if config.StreamDelay < 0 {
		return nil, fmt.Errorf("chat.preview: stream_delay cannot be negative")
	}

	generator := &PreviewGenerator{
		streamDelay: config.StreamDelay,
	}

	if strings.TrimSpace(config.TokenizerSource.Source) == "" {
		return generator, nil
	}

	artifact, err := tokenizer.Load(ctx, config.TokenizerSource)

	if err != nil {
		return nil, err
	}

	generator.artifact = artifact

	return generator, nil
}

/*
Generate streams a deterministic preview response.
*/
func (generator *PreviewGenerator) Generate(
	ctx context.Context,
	prompt string,
	emit func(string) error,
) error {
	response, err := generator.response(prompt)

	if err != nil {
		return err
	}

	return generator.stream(ctx, response, emit)
}

func (generator *PreviewGenerator) response(prompt string) (string, error) {
	if generator.artifact == nil {
		return "Preview runtime active. Use a manifest with system.runtime to run manifest-backed local inference.", nil
	}

	tokenIDs, err := generator.artifact.Tokenizer.Encode(prompt)

	if err != nil {
		return "", err
	}

	return fmt.Sprintf(
		"Preview runtime active. The tokenizer encoded this prompt into %d tokens. Use a manifest with system.runtime to run manifest-backed local inference.",
		len(tokenIDs),
	), nil
}

func (generator *PreviewGenerator) stream(
	ctx context.Context,
	response string,
	emit func(string) error,
) error {
	segments := strings.SplitAfter(response, " ")

	for _, segment := range segments {
		if err := generator.wait(ctx); err != nil {
			return err
		}

		if err := emit(segment); err != nil {
			return err
		}
	}

	return nil
}

func (generator *PreviewGenerator) wait(ctx context.Context) error {
	if generator.streamDelay == 0 {
		return nil
	}

	timer := time.NewTimer(generator.streamDelay)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
