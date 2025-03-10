package process

import (
	"bytes"
	"encoding/json"
	"io"

	"github.com/theapemachine/caramba/pkg/errnie"
)

type PlanData struct {
	Steps []Step `json:"steps" jsonschema:"description=The steps to execute,required"`
}

/*
Plan represents a series of steps to be executed as part of a process.
*/
type Plan struct {
	*PlanData
	enc *json.Encoder
	dec *json.Decoder
	in  *bytes.Buffer
	out *bytes.Buffer
}

/*
Step represents a single action to be taken in a plan.
*/
type Step struct {
	Step string `json:"step" jsonschema:"description=The step to execute,required"`
}

/*
NewPlan creates a new plan with initialized components.
*/
func NewPlan() *Plan {
	errnie.Debug("NewPlan")

	in := bytes.NewBuffer([]byte{})
	out := bytes.NewBuffer([]byte{})

	plan := &Plan{
		PlanData: &PlanData{
			Steps: []Step{},
		},
		enc: json.NewEncoder(out),
		dec: json.NewDecoder(in),
		in:  in,
		out: out,
	}

	// Pre-encode the plan data to JSON for reading
	plan.enc.Encode(plan.PlanData)

	return plan
}

/*
WithSteps adds a new step to the plan.
*/
func (plan *Plan) WithSteps(steps ...Step) *Plan {
	errnie.Debug("Plan.WithSteps")

	plan.PlanData.Steps = append(plan.PlanData.Steps, steps...)
	return plan
}

/*
Read serializes the plan to JSON and writes it to the provided buffer.
*/
func (plan *Plan) Read(buf []byte) (n int, err error) {
	errnie.Debug("Plan.Read")

	if plan.out.Len() == 0 {
		if err = errnie.NewErrIO(plan.enc.Encode(plan.PlanData)); err != nil {
			return 0, err
		}
	}

	if n, err = plan.out.Read(buf); n == 0 {
		return n, io.EOF
	}

	return n, errnie.NewErrIO(err)
}

/*
Write updates the plan from JSON data.
*/
func (plan *Plan) Write(p []byte) (n int, err error) {
	errnie.Debug("Plan.Write", "p", string(p))

	// Reset the output buffer whenever we write new data
	if plan.out.Len() > 0 {
		plan.out.Reset()
	}

	// Write the incoming bytes to the input buffer
	n, err = plan.in.Write(p)
	if err != nil {
		return n, err
	}

	// Try to decode the data from the input buffer
	// If it fails, we still return the bytes written but keep the error
	var buf PlanData
	if decErr := plan.dec.Decode(&buf); decErr == nil {
		// Only update if decoding was successful
		plan.PlanData.Steps = buf.Steps

		// Re-encode to the output buffer for subsequent reads
		if encErr := plan.enc.Encode(plan.PlanData); encErr != nil {
			return n, errnie.NewErrIO(encErr)
		}
	}

	return n, nil
}

/*
Close performs any necessary cleanup.
*/
func (p *Plan) Close() error {
	errnie.Debug("Plan.Close")

	p.PlanData.Steps = nil
	p.in.Reset()
	p.out.Reset()

	return nil
}
