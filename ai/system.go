package ai

import (
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

type System struct {
	Identity *Identity `json:"identity" jsonschema:"title=Identity,description=The identity of the agent,required"`
	subkey   string
}

func NewSystem(identity *Identity, structured bool) *System {
	return &System{
		Identity: identity,
		subkey: func() string {
			if structured {
				return "structured"
			}

			return "unstructured"
		}(),
	}
}

func (system *System) String() string {
	v := viper.GetViper()

	if system.Identity == nil {
		errnie.Error(errors.New("identity is not allowed to be nil"))
		os.Exit(1)
	}

	rolePrompt := v.GetString("prompts.roles." + system.Identity.Role)

	if rolePrompt == "" {
		errnie.Error(errors.New("role prompt is not allowed to be empty"))
		os.Exit(1)
	}

	systemPrompt := v.GetString("prompts.system." + system.subkey)

	if systemPrompt == "" {
		errnie.Error(errors.New("system prompt is not allowed to be empty"))
		os.Exit(1)
	}

	systemPrompt = strings.ReplaceAll(systemPrompt, "{{role}}", rolePrompt)
	systemPrompt = strings.ReplaceAll(systemPrompt, "{{identity}}", system.Identity.String())

	return utils.JoinWith(
		"\n",
		"<system>",
		fmt.Sprintf("\t%s", systemPrompt),
		"</system>",
	)
}
