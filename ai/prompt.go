package ai

import (
	"strings"

	"github.com/spf13/viper"
)

type Prompt struct {
	role   string
	schema string
	name   string
}

func NewPrompt(name, role string, process Process) *Prompt {
	return &Prompt{
		role:   viper.GetString("prompts.roles." + role),
		schema: process.GenerateSchema(),
		name:   name,
	}
}

func (p *Prompt) Build() string {
	base := viper.GetString("prompts.system")

	substitutions := map[string]string{
		"role":   p.role,
		"schema": p.schema,
		"name":   p.name,
	}

	for key, value := range substitutions {
		base = strings.ReplaceAll(base, "{{"+key+"}}", value)
	}

	return base
}
