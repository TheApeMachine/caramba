package tweaker

import (
	"fmt"
	"strings"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/utils"
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
	systemPrompt = strings.ReplaceAll(systemPrompt, "<{tools}>", strings.TrimSpace(utils.Indent(schemas, 1)))

	return systemPrompt
}
