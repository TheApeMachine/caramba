package ai

import (
	"github.com/spf13/viper"
)

type System struct {
	fragments map[string]string
}

func NewSystem(identity *Identity, instructions string, structured bool) *System {
	v := viper.GetViper()
	subkey := "unstructured"

	if structured {
		subkey = "structured"
	}

	return &System{
		fragments: map[string]string{
			"prompt":       v.GetString("prompts.system." + subkey),
			"name":         identity.Name,
			"role":         identity.Role,
			"instructions": v.GetString("prompts.instructions." + instructions),
		},
	}
}
