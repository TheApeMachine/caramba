package provider

import (
	context "context"
	"fmt"

	datura "github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	prvdr "github.com/theapemachine/caramba/pkg/provider"
)

var providers = map[string]prvdr.ProviderType{
	"openai": prvdr.NewOpenAIProvider(),
}

type ProviderRPCServer struct {
	provider *Provider
}

func NewProviderRPCServer(provider *Provider) *ProviderRPCServer {
	return &ProviderRPCServer{provider: provider}
}

func (srv *ProviderRPCServer) Generate(
	ctx context.Context,
	call RPC_generate,
) (err error) {
	errnie.Debug("provider.Generate")

	cfn := errnie.Try(call.Args().Context())
	result := errnie.Try(call.AllocResults())

	out := datura.New(
		datura.WithArtifact(&cfn),
	)

	prvdr, ok := providers["openai"]

	if !ok {
		return errnie.Error(errnie.BadRequest(
			fmt.Errorf("unknown provider: %s", "openai"),
		))
	}

	response := prvdr.Generate(out)

	return result.SetOut(*response.Artifact)
}

func ProviderToClient(provider *Provider) RPC {
	server := NewProviderRPCServer(provider)
	return RPC_ServerToClient(server)
}
