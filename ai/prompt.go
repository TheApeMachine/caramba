package ai

import (
	"strconv"
	"strings"

	"github.com/spf13/viper"
)

/*
Prompt is a wrapper around composable prompts, which use text fragments
defined in the config file.
*/
type Prompt struct {
	role   string
	schema string
	name   string
}

/*
NewPrompt creates a new Prompt instance.
*/
func NewPrompt(name, role string, process Process) *Prompt {
	return &Prompt{
		role:   viper.GetString("prompts.roles." + role),
		schema: process.GenerateSchema(),
		name:   name,
	}
}

/*
Build builds the initial system prompt.
*/
func (prompt *Prompt) Build() string {
	return prompt.applySubstitutions(
		viper.GetString("prompts.system"),
		map[string]string{
			"role":   prompt.role,
			"schema": prompt.schema,
			"name":   prompt.name,
		},
	)
}

/*
BuildTask builds the task prompt.
*/
func (prompt *Prompt) BuildTask(
	task string,
) string {
	return prompt.applySubstitutions(
		viper.GetString("prompts.templates.task"),
		map[string]string{
			"role": prompt.role,
			"task": task,
		},
	)
}

/*
BuildStatus builds the status prompt.
*/
func (prompt *Prompt) BuildStatus(
	iteration int,
	maxIter int,
) string {
	return prompt.applySubstitutions(
		viper.GetString("prompts.templates.task"),
		map[string]string{
			"iteration": strconv.Itoa(iteration),
			"maxIter":   strconv.Itoa(maxIter),
		},
	)
}

/*
applySubstitutions applies the substitutions to a base string.
*/
func (prompt *Prompt) applySubstitutions(
	base string,
	substitutions map[string]string,
) string {
	for key, value := range substitutions {
		base = strings.ReplaceAll(base, "{{"+key+"}}", value)
	}

	return base
}
