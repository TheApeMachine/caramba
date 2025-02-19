package ai

import (
	"fmt"
	"strings"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/utils"
)

/*
Identity defines the Agent that carries the Identity.
It is also the schema that allows Agent to also be a Tool, which allows Agents to build Agents.
*/
type Identity struct {
	ID           string   `json:"id" jsonschema:"description:The unique identifier for the identity,required"`
	Name         string   `json:"name" jsonschema:"description:The name of the identity,required"`
	Description  string   `json:"description" jsonschema:"description:A description of the identity,required"`
	Role         string   `json:"role" jsonschema:"description:The role of the identity,required"`
	Personality  string   `json:"personality" jsonschema:"description:The personality of the identity,required"`
	Motivation   string   `json:"motivation" jsonschema:"description:The motivation of the identity,required"`
	Beliefs      string   `json:"beliefs" jsonschema:"description:The beliefs of the identity,required"`
	Goals        []string `json:"goals" jsonschema:"description:The goals of the identity,required"`
	Instructions string   `json:"instructions" jsonschema:"description:The instructions of the identity,required"`
}

/*
NewIdentity creates a new Identity from the config.
*/
func NewIdentity(configKey string) *Identity {
	v := viper.GetViper()

	return &Identity{
		ID:           v.GetString(fmt.Sprintf("identities.%s.id", configKey)),
		Name:         v.GetString(fmt.Sprintf("identities.%s.name", configKey)),
		Description:  v.GetString(fmt.Sprintf("identities.%s.description", configKey)),
		Role:         v.GetString(fmt.Sprintf("identities.%s.role", configKey)),
		Personality:  v.GetString(fmt.Sprintf("identities.%s.personality", configKey)),
		Motivation:   v.GetString(fmt.Sprintf("identities.%s.motivation", configKey)),
		Beliefs:      v.GetString(fmt.Sprintf("identities.%s.beliefs", configKey)),
		Goals:        v.GetStringSlice(fmt.Sprintf("identities.%s.goals", configKey)),
		Instructions: v.GetString(fmt.Sprintf("identities.%s.instructions", configKey)),
	}
}

/*
Schema return the schema needed when Agent is being used as a Tool.
*/
func (identity *Identity) Schema() interface{} {
	return utils.GenerateSchema[Identity]()
}

/*
String returns the string representation of the identity.
*/
func (identity *Identity) String() string {
	goals := []string{}

	for _, goal := range identity.Goals {
		goals = append(goals, fmt.Sprintf("\t\t<goal>%s</goal>", goal))
	}

	return strings.Join([]string{
		"<identity>",
		fmt.Sprintf("\t<id>%s</id>", identity.ID),
		fmt.Sprintf("\t<name>%s</name>", identity.Name),
		fmt.Sprintf("\t<description>%s</description>", identity.Description),
		fmt.Sprintf("\t<role>%s</role>", identity.Role),
		fmt.Sprintf("\t<personality>%s</personality>", identity.Personality),
		fmt.Sprintf("\t<motivation>%s</motivation>", identity.Motivation),
		fmt.Sprintf("\t<beliefs>%s</beliefs>", identity.Beliefs),
		fmt.Sprintf("\t<goals>\n%s\n\t</goals>", strings.Join(goals, "\n")),
		"</identity>",
	}, "\n")
}

// NewIdentityFromMap creates a new Identity from a map
func NewIdentityFromMap(data map[string]interface{}) *Identity {
	goals := make([]string, 0)
	if g, ok := data["goals"].([]string); ok {
		goals = g
	} else if g, ok := data["goals"].([]interface{}); ok {
		for _, goal := range g {
			if s, ok := goal.(string); ok {
				goals = append(goals, s)
			}
		}
	}

	return &Identity{
		ID:           data["id"].(string),
		Name:         data["name"].(string),
		Description:  data["description"].(string),
		Role:         data["role"].(string),
		Personality:  data["personality"].(string),
		Motivation:   data["motivation"].(string),
		Beliefs:      data["beliefs"].(string),
		Goals:        goals,
		Instructions: data["instructions"].(string),
	}
}
