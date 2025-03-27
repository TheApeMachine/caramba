package provider

import (
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

type MockProvider struct {
	metadata *datura.Artifact
	buffer   *stream.Buffer
	params   *Params
}

var code = []*Message{
	{
		Role: "assistant",
		ToolCalls: []ToolCall{
			{
				Function: ToolCallFunction{
					Name:      "environment",
					Arguments: `{"command":"echo \"import random\n\ndef play_game():\n    print('Welcome to the Number Guessing Game!')\n    number_to_guess = random.randint(1, 100)\n    attempts = 0\n    while True:\n        guess = int(input('Enter your guess (1-100): '))\n        attempts += 1\n        if guess < number_to_guess:\n            print('Too low! Try again.')\n        elif guess > number_to_guess:\n            print('Too high! Try again.')\n        else:\n            print(f'Congratulations! You guessed the number {number_to_guess} in {attempts} attempts.')\n            break\n\nplay_game()\" > number_mock_guessing_game.py"}`,
				},
			},
		},
	},
	{
		Role: "assistant",
		ToolCalls: []ToolCall{
			{
				Function: ToolCallFunction{
					Name:      "environment",
					Arguments: `{"command":"python3 number_mock_guessing_game.py"}`,
				},
			},
		},
	},
	{
		Role: "assistant",
		ToolCalls: []ToolCall{
			{
				Function: ToolCallFunction{
					Name:      "environment",
					Arguments: `{"input":"50\n"}`,
				},
			},
		},
	},
	{
		Role: "assistant",
		ToolCalls: []ToolCall{
			{
				Function: ToolCallFunction{
					Name:      "environment",
					Arguments: `{"input":"10\n"}`,
				},
			},
		},
	},
	{
		Role: "assistant",
		ToolCalls: []ToolCall{
			{
				Function: ToolCallFunction{
					Name:      "environment",
					Arguments: `{"input":"42\n"}`,
				},
			},
		},
	},
}

func NewMockProvider() *MockProvider {
	metadata := datura.New(
		datura.WithRole(datura.ArtifactRoleMetadata),
		datura.WithScope(datura.ArtifactScopeProvider),
	)

	params := &Params{
		Messages: []*Message{},
	}

	return &MockProvider{
		metadata: metadata,
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("provider.MockProvider.Buffer.fn")

			if artifact.Is(datura.ArtifactRoleInspect, datura.ArtifactScopeProvider) {
				// System is requesting our metadata, so we override the buffer artifact.
				*artifact = *metadata
				return
			}

			params.Messages = append(params.Messages, code[len(params.Messages)])

			return artifact.From(params)
		}),
		params: params,
	}
}

func (prvdr *MockProvider) Read(p []byte) (n int, err error) {
	errnie.Debug("provider.MockProvider.Read")
	return prvdr.buffer.Read(p)
}

func (prvdr *MockProvider) Write(p []byte) (n int, err error) {
	errnie.Debug("provider.MockProvider.Write")
	return prvdr.buffer.Write(p)
}

func (prvdr *MockProvider) Close() error {
	errnie.Debug("provider.MockProvider.Close")
	return prvdr.buffer.Close()
}
