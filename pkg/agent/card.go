package agent

import (
	"github.com/theapemachine/caramba/pkg/auth"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

type Card struct {
	Name               string              `json:"name,omitempty"`
	Description        string              `json:"description,omitempty"`
	URL                string              `json:"url,omitempty"`
	Provider           Provider            `json:"provider,omitempty"`
	Version            string              `json:"version,omitempty"`
	Authentication     auth.Authentication `json:"authentication,omitempty"`
	DefaultInputModes  []string            `json:"defaultInputModes,omitempty"`
	DefaultOutputModes []string            `json:"defaultOutputModes,omitempty"`
	Capabilities       Capabilities        `json:"capabilities,omitempty"`
	Skills             []Skill             `json:"skills,omitempty"`
}

func NewCard(
	name string,
	description string,
	url string,
	provider Provider,
	version string,
	authentication auth.Authentication,
	defaultInputModes []string,
	defaultOutputModes []string,
	capabilities Capabilities,
	skills []Skill,
) *Card {
	return &Card{
		Name:               name,
		Description:        description,
		URL:                url,
		Provider:           provider,
		Version:            version,
		Authentication:     authentication,
		DefaultInputModes:  defaultInputModes,
		DefaultOutputModes: defaultOutputModes,
		Capabilities:       capabilities,
	}
}

var settings = "settings.agents."

func FromConfig(name string) *Card {
	return NewCard(
		tweaker.Value[string](settings+name+".name"),
		tweaker.Value[string](settings+name+".description"),
		tweaker.Value[string](settings+name+".url"),
		Provider{
			Organization: tweaker.Value[string](settings + name + ".provider.organization"),
			URL:          tweaker.Value[string](settings + name + ".provider.url"),
		},
		tweaker.Value[string](settings+name+".version"),
		auth.Authentication{
			Schemes: tweaker.Value[string](settings + name + ".authentication.schemes"),
		},
		tweaker.Value[[]string](settings+name+".defaultInputModes"),
		tweaker.Value[[]string](settings+name+".defaultOutputModes"),
		Capabilities{
			Streaming:         tweaker.Value[bool](settings + name + ".capabilities.streaming"),
			PushNotifications: tweaker.Value[bool](settings + name + ".capabilities.pushNotifications"),
		},
		[]Skill{},
	)
}

type Provider struct {
	Organization string `json:"organization,omitempty"`
	URL          string `json:"url,omitempty"`
}

type Capabilities struct {
	Streaming         bool `json:"streaming,omitempty"`
	PushNotifications bool `json:"pushNotifications,omitempty"`
}

type Skill struct {
	ID          string   `json:"id,omitempty"`
	Name        string   `json:"name,omitempty"`
	Description string   `json:"description,omitempty"`
	Tags        []string `json:"tags,omitempty"`
	Examples    []string `json:"examples,omitempty"`
}
