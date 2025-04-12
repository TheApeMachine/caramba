package agent

import (
	"github.com/theapemachine/caramba/pkg/auth"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

type Card struct {
	Name               string              `json:"name,omitempty"`
	Description        string              `json:"description,omitempty"`
	URL                string              `json:"url,omitempty"`
	Provider           Provider            `json:"provider"`
	Version            string              `json:"version,omitempty"`
	DocumentationURL   string              `json:"documentationURL,omitempty"`
	Authentication     auth.Authentication `json:"authentication"`
	DefaultInputModes  []string            `json:"defaultInputModes,omitempty"`
	DefaultOutputModes []string            `json:"defaultOutputModes,omitempty"`
	Capabilities       Capabilities        `json:"capabilities"`
	Skills             []Skill             `json:"skills,omitempty"`
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
			Schemes: tweaker.Value[string](settings + name + ".authentication.schemes"),
		},
		DefaultInputModes:  tweaker.Value[[]string](settings + name + ".defaultInputModes"),
		DefaultOutputModes: tweaker.Value[[]string](settings + name + ".defaultOutputModes"),
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
	Streaming              bool `json:"streaming,omitempty"`
	PushNotifications      bool `json:"pushNotifications,omitempty"`
	StateTransitionHistory bool `json:"stateTransitionHistory,omitempty"`
}

type Skill struct {
	ID          string   `json:"id,omitempty"`
	Name        string   `json:"name,omitempty"`
	Description string   `json:"description,omitempty"`
	Tags        []string `json:"tags,omitempty"`
	Examples    []string `json:"examples,omitempty"`
	InputModes  []string `json:"inputModes,omitempty"`
	OutputModes []string `json:"outputModes,omitempty"`
}
