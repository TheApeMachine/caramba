package task

// Artifact represents an artifact in the A2A protocol
type Artifact struct {
	Name        *string                `json:"name,omitempty"`
	Description *string                `json:"description,omitempty"`
	Parts       []Part                 `json:"parts"`
	Index       int                    `json:"index"`
	Append      *bool                  `json:"append,omitempty"`
	LastChunk   *bool                  `json:"lastChunk,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}
