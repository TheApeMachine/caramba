package tweaker

import (
	"fmt"
	"strings"

	"github.com/spf13/viper"
)

func GetSystemPrompt(system, name, role, schemas string) string {
	v := viper.GetViper()

	systemPrompt := v.GetString(
		fmt.Sprintf("prompts.templates.systems.%s", system),
	)

	if systemPrompt == "" {
		systemPrompt = system
	}

	systemPrompt = strings.ReplaceAll(systemPrompt, "<{role}>", v.GetString(fmt.Sprintf("prompts.templates.roles.%s", role)))
	systemPrompt = strings.ReplaceAll(systemPrompt, "<{identity}>", "NAME: "+name)
	systemPrompt = strings.ReplaceAll(systemPrompt, "<{tools}>", indent(schemas, 1))

	return systemPrompt
}

func indent(text string, index int) string {
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		lines[i] = strings.Repeat("  ", index) + line
	}
	return strings.Join(lines, "\n")
}
