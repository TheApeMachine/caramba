package tweaker

import (
	"strconv"
	"strings"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/utils"
)

func GetUserPrompt(user string) string {
	v := viper.GetViper()

	userPrompt := v.GetString("prompts.templates.user")
	userPrompt = strings.ReplaceAll(userPrompt, "<{user}>", "> "+user)

	return "\n" + userPrompt + "\n"
}

func GetContext() string {
	v := viper.GetViper()
	return v.GetString("prompts.templates.context")
}

func GetIteration(name, role string, iteration int, response string) string {
	v := viper.GetViper()

	iterationPrompt := v.GetString("prompts.templates.iteration")
	iterationPrompt = strings.ReplaceAll(iterationPrompt, "<{agent}>", name)
	iterationPrompt = strings.ReplaceAll(iterationPrompt, "<{role}>", role)
	iterationPrompt = strings.ReplaceAll(iterationPrompt, "<{iteration}>", strconv.Itoa(iteration))
	iterationPrompt = strings.ReplaceAll(iterationPrompt, "<{response}>", utils.Indent(response, 1))

	return strings.TrimSpace(iterationPrompt)
}
