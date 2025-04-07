package provider

import (
	"context"

	"github.com/theapemachine/caramba/pkg/datura"
)

type ProviderType interface {
	Generate(context.Context, *datura.ArtifactBuilder) *datura.ArtifactBuilder
}

type Message struct {
	Role    string `json:"role"`
	Name    string `json:"name"`
	Content string `json:"content"`
}
