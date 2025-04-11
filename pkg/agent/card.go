package agent

type Card struct {
	Name               string         `json:"name,omitempty"`
	Description        string         `json:"description,omitempty"`
	URL                string         `json:"url,omitempty"`
	Provider           Provider       `json:"provider,omitempty"`
	Version            string         `json:"version,omitempty"`
	Authentication     Authentication `json:"authentication,omitempty"`
	DefaultInputModes  []string       `json:"defaultInputModes,omitempty"`
	DefaultOutputModes []string       `json:"defaultOutputModes,omitempty"`
	Capabilities       Capabilities   `json:"capabilities,omitempty"`
	Skills             []Skill        `json:"skills,omitempty"`
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

// MessagePart represents a part of a message that can be text, file, or data
type MessagePart interface {
	GetType() string
}
