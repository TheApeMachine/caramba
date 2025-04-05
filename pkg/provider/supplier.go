package provider

import (
	"github.com/theapemachine/caramba/pkg/datura"
)

type ProviderType interface {
	Generate(*datura.ArtifactBuilder) *datura.ArtifactBuilder
}

type Message struct {
	Role    string `json:"role"`
	Name    string `json:"name"`
	Content string `json:"content"`
}
