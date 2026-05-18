package compiler

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/runtime/program"
)

/*
compileSteps reads system.runtime.program — the ordered list of
runtime IR steps — and compiles each entry. Both the top-level
program key and the `steps` alias are accepted so manifests can use
whichever reads better.
*/
func compileSteps(runtimeBlock map[string]any, runtimeProgram *program.Program) error {
	raw, ok := runtimeBlock["program"]

	if !ok {
		raw = runtimeBlock["steps"]
	}

	list, err := yamlList(raw, "system.runtime.program")

	if err != nil {
		return err
	}

	for index, entry := range list {
		step, err := compileStep(entry, fmt.Sprintf("program[%d]", index))

		if err != nil {
			return err
		}

		runtimeProgram.Steps = append(runtimeProgram.Steps, step)
	}

	return nil
}

/*
compileStep turns a single YAML step entry into program.Step. It
recognizes the shorthand top-level keys the platform requirements
document uses (graph, sampler, scheduler, tokenizer, count, as,
source, condition) and folds them into Config / Inputs as the
canonical fields.
*/
func compileStep(entry any, where string) (program.Step, error) {
	body, err := yamlMap(entry, where)

	if err != nil {
		return program.Step{}, err
	}

	if body == nil {
		return program.Step{}, fmt.Errorf("compiler: %s is empty", where)
	}

	stepID, err := yamlString(body["id"], where+".id")

	if err != nil {
		return program.Step{}, err
	}

	opName, err := yamlString(body["op"], where+".op")

	if err != nil {
		return program.Step{}, err
	}

	step := program.Step{
		ID:      stepID,
		Op:      program.OperationID(opName),
		Inputs:  map[string]program.ValueRef{},
		Outputs: map[string]program.ValueRef{},
		Config:  map[string]any{},
	}

	if err := mergeInputsMap(step.Inputs, body["inputs"], where+".inputs"); err != nil {
		return program.Step{}, err
	}

	if err := mergeOutputsMap(step.Outputs, body["outputs"], where+".outputs"); err != nil {
		return program.Step{}, err
	}

	if err := mergeConfigMap(step.Config, body["config"], where+".config"); err != nil {
		return program.Step{}, err
	}

	for key, value := range body {
		if isReservedStepKey(key) {
			continue
		}

		if err := applyShorthand(&step, key, value, where); err != nil {
			return program.Step{}, err
		}
	}

	if body["body"] != nil {
		bodyList, err := yamlList(body["body"], where+".body")

		if err != nil {
			return program.Step{}, err
		}

		for childIndex, childEntry := range bodyList {
			childStep, err := compileStep(
				childEntry, fmt.Sprintf("%s.body[%d]", where, childIndex),
			)

			if err != nil {
				return program.Step{}, err
			}

			step.Body = append(step.Body, childStep)
		}
	}

	return step, nil
}

var reservedStepKeys = map[string]bool{
	"id":      true,
	"op":      true,
	"inputs":  true,
	"outputs": true,
	"config":  true,
	"body":    true,
}

func isReservedStepKey(key string) bool {
	return reservedStepKeys[key]
}

/*
applyShorthand handles the inline forms used throughout the platform
requirements document. Every shorthand form maps to a canonical Inputs /
Outputs / Config slot so a downstream tool that walks Inputs always
finds the same shape regardless of the YAML the author wrote.
*/
func applyShorthand(step *program.Step, key string, value any, where string) error {
	switch key {
	case "graph":
		return setNamespaceInput(step, "graph", program.NamespaceGraph, value, where)
	case "sampler":
		return setNamespaceInput(step, "sampler", program.NamespaceSampler, value, where)
	case "scheduler":
		return setNamespaceInput(step, "scheduler", program.NamespaceScheduler, value, where)
	case "tokenizer":
		return setNamespaceInput(step, "tokenizer", program.NamespaceTokenizer, value, where)
	case "count", "as", "update", "delta", "max", "start", "end":
		step.Config[key] = value

		return nil
	case "source", "condition", "text", "token", "tokens", "logits", "history",
		"latents", "velocity", "timestep", "step_index", "element", "value",
		"stream", "image":
		ref, err := parseInputRef(value, where+"."+key)

		if err != nil {
			return err
		}

		step.Inputs[key] = ref

		return nil
	case "target", "report":
		ref, err := parseInputRef(value, where+"."+key)

		if err != nil {
			return err
		}

		step.Outputs[key] = ref

		return nil
	}

	step.Config[key] = value

	return nil
}

func setNamespaceInput(
	step *program.Step, key, namespace string, value any, where string,
) error {
	name, err := yamlString(value, where+"."+key)

	if err != nil {
		return err
	}

	if name == "" {
		return fmt.Errorf("compiler: %s.%s requires a name", where, key)
	}

	step.Inputs[key] = program.ValueRef{Namespace: namespace, Name: name}

	return nil
}

func mergeInputsMap(target map[string]program.ValueRef, value any, where string) error {
	entries, err := yamlMap(value, where)

	if err != nil {
		return err
	}

	for name, raw := range entries {
		ref, err := parseInputRef(raw, where+"."+name)

		if err != nil {
			return err
		}

		target[name] = ref
	}

	return nil
}

func mergeOutputsMap(target map[string]program.ValueRef, value any, where string) error {
	entries, err := yamlMap(value, where)

	if err != nil {
		return err
	}

	for name, raw := range entries {
		text, err := yamlString(raw, where+"."+name)

		if err != nil {
			return err
		}

		ref, err := program.ParseValueRef(text)

		if err != nil {
			return fmt.Errorf("compiler: %s.%s: %w", where, name, err)
		}

		target[name] = ref
	}

	return nil
}

func mergeConfigMap(target map[string]any, value any, where string) error {
	entries, err := yamlMap(value, where)

	if err != nil {
		return err
	}

	for key, raw := range entries {
		target[key] = raw
	}

	return nil
}

func parseInputRef(value any, where string) (program.ValueRef, error) {
	text, err := yamlString(value, where)

	if err != nil {
		return program.ValueRef{}, err
	}

	if text == "" {
		return program.ValueRef{}, fmt.Errorf("compiler: %s is empty", where)
	}

	ref, err := program.ParseValueRef(text)

	if err != nil {
		return program.ValueRef{}, fmt.Errorf("compiler: %s: %w", where, err)
	}

	return ref, nil
}
