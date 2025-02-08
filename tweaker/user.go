package tweaker

import (
	"strconv"
	"strings"

	"github.com/spf13/viper"
)

func GetUserPrompt(user string) string {
	v := viper.GetViper()

	userPrompt := v.GetString("prompts.templates.user")
	userPrompt = strings.ReplaceAll(userPrompt, "<{user}>", indent("> "+user, 1))

	return "\n" + userPrompt
}

func GetContext() string {
	v := viper.GetViper()
	return "\n" + v.GetString("prompts.templates.context") + "\n"
}

func GetIteration(name, role string, iteration int, response string) string {
	v := viper.GetViper()

	iterationPrompt := v.GetString("prompts.templates.iteration")
	iterationPrompt = strings.ReplaceAll(iterationPrompt, "<{agent}>", name)
	iterationPrompt = strings.ReplaceAll(iterationPrompt, "<{role}>", role)
	iterationPrompt = strings.ReplaceAll(iterationPrompt, "<{iteration}>", strconv.Itoa(iteration))
	iterationPrompt = strings.ReplaceAll(iterationPrompt, "<{response}>", indent(response, 1))

	return "\n" + iterationPrompt
}
