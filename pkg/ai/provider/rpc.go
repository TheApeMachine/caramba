package provider

import (
	context "context"

	"github.com/theapemachine/caramba/pkg/errnie"
	aiprvdr "github.com/theapemachine/caramba/pkg/provider"
)

type ProviderRPCServer struct {
	provider Provider
}

func NewProviderRPCServer(
	provider Provider,
) *ProviderRPCServer {
	errnie.Trace("provider.NewProviderRPCServer")

	return &ProviderRPCServer{
		provider: provider,
	}
}

func (srv *ProviderRPCServer) Generate(
	ctx context.Context,
	call RPC_generate,
) (err error) {
	errnie.Trace("provider.Generate RPC", "provider_name", srv.provider.Name)

	artifact := errnie.Try(call.Args().Artifact())
	result := errnie.Try(call.AllocResults())

	response := getProvider(errnie.Try(srv.provider.Name())).Generate(ctx, &artifact)
	return result.SetOut(*response)
}

func ProviderToClient(
	provider Provider,
) RPC {
	errnie.Trace("provider.ProviderToClient")

	server := NewProviderRPCServer(provider)
	return RPC_ServerToClient(server)
}

func getProvider(name string) aiprvdr.ProviderType {
	errnie.Trace("provider.getProvider", "name", name)

	switch name {
	case "openai":
		return aiprvdr.NewOpenAIProvider()
	}

	return nil
}
