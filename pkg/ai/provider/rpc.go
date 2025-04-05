package provider

import (
	context "context"

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
	cfn := errnie.Try(call.Args().Context())
	result := errnie.Try(call.AllocResults())

	name := errnie.Try(Provider(cfn).Name())
	prvdr := providers[name]

	builder := datura.New()
	builder.Artifact = &cfn
	response := prvdr.Generate(builder)

	return result.SetOut(*response.Artifact)
}

func ProviderToClient(provider *Provider) RPC {
	server := NewProviderRPCServer(provider)
	return RPC_ServerToClient(server)
}
