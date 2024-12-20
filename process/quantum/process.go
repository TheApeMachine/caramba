package quantum

import (
	"github.com/theapemachine/amsh/utils"
)

/*
Process represents a quantum-layered thought or reasoning process.
*/
type Process struct {
	SuperpositionStates []SuperpositionState `json:"superposition_states" jsonschema:"required,title=SuperpositionStates,description=Multiple simultaneous possibilities"`
	Entanglements       []Entanglement       `json:"entanglements" jsonschema:"required,title=Entanglements,description=Correlated state relationships"`
	WaveFunction        WaveFunction         `json:"wave_function" jsonschema:"required,title=WaveFunction,description=Probability distribution of states"`
}

// ComplexNumber represents a complex number in JSON-serializable format
type ComplexNumber struct {
	Real      float64 `json:"real" jsonschema:"required,title=Real,description=Real part of the complex number"`
	Imaginary float64 `json:"imaginary" jsonschema:"required,title=Imaginary,description=Imaginary part of the complex number"`
}

// WaveFunction now uses JSON-serializable complex numbers
type WaveFunction struct {
	StateSpaceDim int             `json:"state_space_dim" jsonschema:"required,title=StateSpaceDim,description=Dimension of the state space"`
	Amplitudes    []ComplexNumber `json:"amplitudes" jsonschema:"required,title=Amplitudes,description=Quantum state amplitudes"`
	Basis         []string        `json:"basis" jsonschema:"required,title=Basis,description=Names of basis states"`
	Time          string          `json:"time" jsonschema:"required,title=Time,description=Time of the wave function"`
}

type SuperpositionState struct {
	ID            string             `json:"id" jsonschema:"required,title=ID,description=Unique identifier for the state"`
	Possibilities map[string]float64 `json:"possibilities" jsonschema:"required,title=Possibilities,description=Possible states and probabilities"`
	Phase         float64            `json:"phase" jsonschema:"description=Quantum phase"`
	Coherence     float64            `json:"coherence" jsonschema:"required,title=Coherence,description=State coherence"`
	Lifetime      string             `json:"lifetime" jsonschema:"required,title=Lifetime,description=Expected lifetime"`
}

type Entanglement struct {
	ID       string   `json:"id" jsonschema:"required,title=ID,description=Unique identifier for entanglement"`
	StateIDs []string `json:"state_ids" jsonschema:"required,title=StateIDs,description=IDs of entangled states"`
	Strength float64  `json:"strength" jsonschema:"required,title=Strength,description=Entanglement strength"`
	Type     string   `json:"type" jsonschema:"required,title=Type,description=Type of entanglement"`
	Duration string   `json:"duration" jsonschema:"required,title=Duration,description=Expected duration"`
}

func (proc *Process) GenerateSchema() string {
	return utils.GenerateSchema[Process]()
}
