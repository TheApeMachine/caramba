package holographic

import (
	"github.com/theapemachine/amsh/utils"
)

/*
Process represents distributed information storage.
*/
type Process struct {
	Encodings          []Encoding        `json:"encodings" jsonschema:"description=Distributed information patterns,required"`
	InterferenceSpace  InterferenceSpace `json:"interference_space" jsonschema:"description=Interaction between encodings,required"`
	ReconstructionKeys []string          `json:"reconstruction_keys" jsonschema:"description=Access patterns for information retrieval,required"`
}

type Encoding struct {
	ID       string    `json:"id" jsonschema:"required,description=Unique identifier for the encoding"`
	Pattern  []float64 `json:"pattern" jsonschema:"required,description=Pattern of the encoding"`
	Phase    float64   `json:"phase" jsonschema:"required,description=Phase of the encoding"`
	Position []int     `json:"position" jsonschema:"required,description=Position of the encoding"`
	Strength float64   `json:"strength" jsonschema:"required,description=Strength of the encoding"`
}

type InterferenceSpace struct {
	Dimensions []int       `json:"dimensions" jsonschema:"required,description=Dimensions of the interference space"`
	Field      []float64   `json:"field" jsonschema:"required,description=Field of the interference space"`
	Resonances []Resonance `json:"resonances" jsonschema:"required,description=Resonances in the interference space"`
	Energy     float64     `json:"energy" jsonschema:"required,description=Energy of the interference space"`
}

type Resonance struct {
	Position  []int   `json:"position" jsonschema:"required,description=Position of the resonance"`
	Strength  float64 `json:"strength" jsonschema:"required,description=Strength of the resonance"`
	Phase     float64 `json:"phase" jsonschema:"required,description=Phase of the resonance"`
	Stability float64 `json:"stability" jsonschema:"required,description=Stability of the resonance"`
}

func (ta *Process) GenerateSchema() string {
	return utils.GenerateSchema[Process]()
}
