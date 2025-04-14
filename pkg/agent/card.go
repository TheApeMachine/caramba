package agent

import (
	"github.com/theapemachine/caramba/pkg/auth"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

type Card struct {
	Name               string              `json:"name,omitempty" redis:"name"`
	Description        string              `json:"description,omitempty" redis:"description"`
	URL                string              `json:"url,omitempty" redis:"url"`
	Provider           Provider            `json:"provider" redis:"provider"`
	Version            string              `json:"version,omitempty" redis:"version"`
	DocumentationURL   string              `json:"documentationURL,omitempty" redis:"documentationURL"`
	Authentication     auth.Authentication `json:"authentication" redis:"authentication"`
	DefaultInputModes  []string            `json:"defaultInputModes,omitempty" redis:"defaultInputModes"`
	DefaultOutputModes []string            `json:"defaultOutputModes,omitempty" redis:"defaultOutputModes"`
	Capabilities       Capabilities        `json:"capabilities" redis:"capabilities"`
	Skills             []Skill             `json:"skills,omitempty" redis:"skills"`
}

func NewCard() *Card {
	return &Card{}
}

var settings = "agents."

func FromConfig(name string) *Card {
	return &Card{
		Name:        tweaker.Value[string](settings + name + ".name"),
		Description: tweaker.Value[string](settings + name + ".description"),
		URL:         tweaker.Value[string](settings + name + ".url"),
		Provider: Provider{
			Organization: tweaker.Value[string](settings + name + ".provider.organization"),
			URL:          tweaker.Value[string](settings + name + ".provider.url"),
		},
		Version: tweaker.Value[string](settings + name + ".version"),
		Authentication: auth.Authentication{
			Schemes: tweaker.GetStringSlice(settings + name + ".authentication.schemes"),
		},
		DefaultInputModes:  tweaker.GetStringSlice(settings + name + ".defaultInputModes"),
		DefaultOutputModes: tweaker.GetStringSlice(settings + name + ".defaultOutputModes"),
		Capabilities: Capabilities{
			Streaming:         tweaker.Value[bool](settings + name + ".capabilities.streaming"),
			PushNotifications: tweaker.Value[bool](settings + name + ".capabilities.pushNotifications"),
		},
		Skills: []Skill{},
	}
}

type Provider struct {
	Organization string `json:"organization,omitempty"`
	URL          string `json:"url,omitempty"`
}

type Capabilities struct {
	Streaming              bool `json:"streaming,omitempty" redis:"streaming"`
	PushNotifications      bool `json:"pushNotifications,omitempty" redis:"pushNotifications"`
	StateTransitionHistory bool `json:"stateTransitionHistory,omitempty" redis:"stateTransitionHistory"`
}

type Skill struct {
	ID          string   `json:"id,omitempty" redis:"id"`
	Name        string   `json:"name,omitempty" redis:"name"`
	Description string   `json:"description,omitempty" redis:"description"`
	Tags        []string `json:"tags,omitempty" redis:"tags"`
	Examples    []string `json:"examples,omitempty" redis:"examples"`
	InputModes  []string `json:"inputModes,omitempty" redis:"inputModes"`
	OutputModes []string `json:"outputModes,omitempty" redis:"outputModes"`
}
