package ai

import (
	"errors"
	"fmt"
	"os"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
)

type System struct {
	Identity *Identity `json:"identity" jsonschema:"title=Identity,description=The identity of the agent,required"`
}

func NewSystem(identity *Identity) *System {
	return &System{
		Identity: identity,
	}
}

func (system *System) String() string {
	v := viper.GetViper()

	systemPrompt := v.GetString("prompts.system")
	rolePrompt := v.GetString("prompts.roles." + system.Identity.Role)

	if systemPrompt == "" {
		errnie.Error(errors.New("system prompt is not allowed to be empty"))
		os.Exit(1)
	}

	if rolePrompt == "" {
		errnie.Error(errors.New("role prompt is not allowed to be empty"))
		os.Exit(1)
	}

	return utils.JoinWith(
		"\n",
		"<system>",
		utils.JoinWith("\n\n",
			fmt.Sprintf("\t%s", systemPrompt),
			system.Identity.String(),
			utils.JoinWith("\n",
				"\t<role>",
				fmt.Sprintf("\t\t%s", rolePrompt),
				"\t</role>",
			),
		),
		"</system>",
	)
}
