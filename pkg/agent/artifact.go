package agent

type Artifact struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parts       []Part         `json:"parts"`
	Metadata    map[string]any `json:"metadata"`
	Index       int            `json:"index"`
	Append      bool           `json:"append"`
	LastChunk   bool           `json:"lastChunk"`
}
