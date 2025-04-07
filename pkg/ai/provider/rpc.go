package provider

import (
	context "context"

	datura "github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type ProviderRPCServer struct {
	provider *ProviderBuilder
}

func NewProviderRPCServer(provider *ProviderBuilder) *ProviderRPCServer {
	return &ProviderRPCServer{provider: provider}
}

func (srv *ProviderRPCServer) Generate(
	ctx context.Context,
	call RPC_generate,
) (err error) {
	errnie.Debug("provider.Generate RPC", "provider_name", srv.provider.Name)

	artifact := errnie.Try(call.Args().Artifact())
	result := errnie.Try(call.AllocResults())

	artifactBuilder := datura.New(
		datura.WithArtifact(&artifact),
	)

	responseBuilder := srv.provider.AIProvider.Generate(ctx, artifactBuilder)

	return result.SetOut(*responseBuilder.Artifact)
}

func ProviderToClient(provider *ProviderBuilder) RPC {
	server := NewProviderRPCServer(provider)
	return RPC_ServerToClient(server)
}
