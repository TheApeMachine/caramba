package ai

import (
	"strings"

	"github.com/spf13/viper"
)

type Prompt struct {
	role   string
	schema string
}

func NewPrompt() *Prompt {
	return &Prompt{}
}

func (p *Prompt) WithRole(role string) *Prompt {
	p.role = viper.GetString("prompts.roles." + role)
	return p
}

func (p *Prompt) WithSchema(schema string) *Prompt {
	p.schema = schema
	return p
}

func (p *Prompt) Build() string {
	base := viper.GetString("prompts.system")

	substitutions := map[string]string{
		"role":   p.role,
		"schema": p.schema,
	}

	for key, value := range substitutions {
		base = strings.ReplaceAll(base, "{{"+key+"}}", value)
	}

	return base
}
