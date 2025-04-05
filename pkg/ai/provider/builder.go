package provider

import (
	"context"

	"capnproto.org/go/capnp/v3"
	aicontext "github.com/theapemachine/caramba/pkg/ai/context"
	tool "github.com/theapemachine/caramba/pkg/ai/tool"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	llm "github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tools"
)

type ProviderBuilder struct {
	*Provider
}

type ProviderBuilderOption func(*ProviderBuilder)

// New creates a new provider with the given name
func New(opts ...ProviderBuilderOption) *ProviderBuilder {
	var (
		arena = capnp.SingleSegment(nil)
		seg   *capnp.Segment
		prvdr Provider
		err   error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return nil
	}

	if prvdr, err = NewRootProvider(seg); errnie.Error(err) != nil {
		return nil
	}

	builder := &ProviderBuilder{
		Provider: &prvdr,
	}

	for _, opt := range opts {
		opt(builder)
	}

	return builder
}

// Generate implements the Provider interface
func (prvdr *Provider) Generate(
	ctx context.Context,
	input string,
) chan *datura.ArtifactBuilder {
	out := make(chan *datura.ArtifactBuilder)

	go func() {
		defer close(out)

		// Create new segment
		_, seg, err := capnp.NewMessage(capnp.SingleSegment(nil))
		if err != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		// Create context with message
		context, err := aicontext.NewRootContext(seg)
		if err != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		// Add input message
		msgs, err := context.NewMessages(1)
		if err != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}
		msg0 := msgs.At(0)
		if err := msg0.SetRole("user"); err != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}
		if err := msg0.SetContent(input); err != nil {
			out <- datura.New(datura.WithError(errnie.Error(err)))
			return
		}

		// Process through provider
		out <- datura.New(datura.WithPayload([]byte(input)))
	}()

	return out
}

// Provider implements the Generate_Server interface
func (prvdr *Provider) Call(ctx context.Context, call Generate_call) error {
	params, err := call.Args().Params()
	if err != nil {
		return errnie.Error(err)
	}

	context, err := call.Args().Context()
	if err != nil {
		return errnie.Error(err)
	}

	toolList, err := call.Args().Tools()
	if err != nil {
		return errnie.Error(err)
	}

	// Convert tool.Tool_List to []mcp.Tool
	var mcpTools []tools.ToolType
	for i := 0; i < toolList.Len(); i++ {
		t := toolList.At(i)

		name, err := t.Name()

		if err != nil {
			continue
		}

		builder := tool.New(tool.WithName(name))

		if builder != nil {
			mcpTools = append(mcpTools, builder.ToMCP().MCP...)
		}
	}

	// Create OpenAI provider
	openaiProvider := llm.NewOpenAIProvider()

	// Generate response using OpenAI
	ch := openaiProvider.Generate(params, context, mcpTools)
	result := <-ch

	// Get response payload
	payload, err := result.DecryptPayload()
	if err != nil {
		return errnie.Error(err)
	}

	// Set response
	results, err := call.AllocResults()
	if err != nil {
		return errnie.Error(err)
	}
	return results.SetOut(string(payload))
}

func (prvdr *ProviderBuilder) Client() Generate {
	return Generate_ServerToClient(prvdr.Provider)
}

func WithName(name string) ProviderBuilderOption {
	return func(p *ProviderBuilder) {
		if err := p.SetName(name); err != nil {
			errnie.Error(err)
		}
	}
}
