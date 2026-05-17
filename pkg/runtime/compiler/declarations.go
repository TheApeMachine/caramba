package compiler

import (
	"fmt"
	"sort"

	"github.com/theapemachine/caramba/pkg/runtime/program"
)

/*
compileAssets reads system.runtime.assets — a map keyed by asset id
where each value contains a `source` and an optional `kind`.
*/
func compileAssets(runtimeBlock map[string]any, runtimeProgram *program.Program) error {
	assets, err := yamlMap(runtimeBlock["assets"], "system.runtime.assets")

	if err != nil {
		return err
	}

	for _, assetID := range sortedKeys(assets) {
		body, err := yamlMap(assets[assetID], fmt.Sprintf("assets.%s", assetID))

		if err != nil {
			return err
		}

		declaration, err := compileAssetBody(assetID, body)

		if err != nil {
			return err
		}

		runtimeProgram.Assets = append(runtimeProgram.Assets, declaration)
	}

	return nil
}

func compileAssetBody(assetID string, body map[string]any) (program.AssetDeclaration, error) {
	declaration := program.AssetDeclaration{ID: assetID, Config: map[string]any{}}
	source, err := yamlString(body["source"], fmt.Sprintf("assets.%s.source", assetID))

	if err != nil {
		return program.AssetDeclaration{}, err
	}

	declaration.Source = source

	kind, err := yamlString(body["kind"], fmt.Sprintf("assets.%s.kind", assetID))

	if err != nil {
		return program.AssetDeclaration{}, err
	}

	declaration.Kind = kind

	if declaration.Kind == "" {
		declaration.Kind = inferKind(assetID)
	}

	for key, value := range body {
		if key == "source" || key == "kind" {
			continue
		}

		declaration.Config[key] = value
	}

	return declaration, nil
}

func inferKind(assetID string) string {
	switch assetID {
	case "tokenizer":
		return "tokenizer"
	case "dataset":
		return "dataset"
	}

	return "model"
}

/*
compileState reads system.runtime.state — a map keyed by state id
where each value contains a `type` and arbitrary config.
*/
func compileState(runtimeBlock map[string]any, runtimeProgram *program.Program) error {
	stateBlock, err := yamlMap(runtimeBlock["state"], "system.runtime.state")

	if err != nil {
		return err
	}

	for _, stateID := range sortedKeys(stateBlock) {
		body, err := yamlMap(stateBlock[stateID], fmt.Sprintf("state.%s", stateID))

		if err != nil {
			return err
		}

		declaration, err := compileStateBody(stateID, body)

		if err != nil {
			return err
		}

		runtimeProgram.State = append(runtimeProgram.State, declaration)
	}

	return nil
}

func compileStateBody(stateID string, body map[string]any) (program.StateDeclaration, error) {
	declaration := program.StateDeclaration{ID: stateID, Config: map[string]any{}}
	typeName, err := yamlString(body["type"], fmt.Sprintf("state.%s.type", stateID))

	if err != nil {
		return program.StateDeclaration{}, err
	}

	declaration.Type = typeName

	backend, err := yamlString(body["backend"], fmt.Sprintf("state.%s.backend", stateID))

	if err != nil {
		return program.StateDeclaration{}, err
	}

	declaration.Backend = backend

	for key, value := range body {
		if key == "type" || key == "backend" {
			continue
		}

		declaration.Config[key] = value
	}

	return declaration, nil
}

/*
compileSamplers reads system.runtime.samplers. Each entry has a
`type` and arbitrary config (temperature, top_k, top_p, etc.).
*/
func compileSamplers(runtimeBlock map[string]any, runtimeProgram *program.Program) error {
	samplerBlock, err := yamlMap(runtimeBlock["samplers"], "system.runtime.samplers")

	if err != nil {
		return err
	}

	for _, samplerID := range sortedKeys(samplerBlock) {
		body, err := yamlMap(samplerBlock[samplerID], fmt.Sprintf("samplers.%s", samplerID))

		if err != nil {
			return err
		}

		declaration, err := compileSamplerBody(samplerID, body)

		if err != nil {
			return err
		}

		runtimeProgram.Samplers = append(runtimeProgram.Samplers, declaration)
	}

	return nil
}

func compileSamplerBody(samplerID string, body map[string]any) (program.SamplerDeclaration, error) {
	declaration := program.SamplerDeclaration{ID: samplerID, Config: map[string]any{}}
	typeName, err := yamlString(body["type"], fmt.Sprintf("samplers.%s.type", samplerID))

	if err != nil {
		return program.SamplerDeclaration{}, err
	}

	declaration.Type = typeName

	for key, value := range body {
		if key == "type" {
			continue
		}

		declaration.Config[key] = value
	}

	return declaration, nil
}

/*
compileSchedulers reads system.runtime.schedulers. Same shape as
samplers.
*/
func compileSchedulers(runtimeBlock map[string]any, runtimeProgram *program.Program) error {
	schedulerBlock, err := yamlMap(runtimeBlock["schedulers"], "system.runtime.schedulers")

	if err != nil {
		return err
	}

	for _, schedulerID := range sortedKeys(schedulerBlock) {
		body, err := yamlMap(schedulerBlock[schedulerID], fmt.Sprintf("schedulers.%s", schedulerID))

		if err != nil {
			return err
		}

		declaration, err := compileSchedulerBody(schedulerID, body)

		if err != nil {
			return err
		}

		runtimeProgram.Schedulers = append(runtimeProgram.Schedulers, declaration)
	}

	return nil
}

func compileSchedulerBody(
	schedulerID string, body map[string]any,
) (program.SchedulerDeclaration, error) {
	declaration := program.SchedulerDeclaration{ID: schedulerID, Config: map[string]any{}}
	typeName, err := yamlString(body["type"], fmt.Sprintf("schedulers.%s.type", schedulerID))

	if err != nil {
		return program.SchedulerDeclaration{}, err
	}

	declaration.Type = typeName

	for key, value := range body {
		if key == "type" {
			continue
		}

		declaration.Config[key] = value
	}

	return declaration, nil
}

/*
compileGraphs reads system.runtime.graphs. Each entry has a
`topology` or `manifest`, plus optional `weight_asset` and arbitrary
config.
*/
func compileGraphs(runtimeBlock map[string]any, runtimeProgram *program.Program) error {
	graphBlock, err := yamlMap(runtimeBlock["graphs"], "system.runtime.graphs")

	if err != nil {
		return err
	}

	for _, graphID := range sortedKeys(graphBlock) {
		body, err := yamlMap(graphBlock[graphID], fmt.Sprintf("graphs.%s", graphID))

		if err != nil {
			return err
		}

		module, err := compileGraphBody(graphID, body)

		if err != nil {
			return err
		}

		runtimeProgram.Graphs[graphID] = module
	}

	return nil
}

func compileGraphBody(graphID string, body map[string]any) (program.GraphModule, error) {
	module := program.GraphModule{ID: graphID, Config: map[string]any{}}
	topology, err := yamlString(body["topology"], fmt.Sprintf("graphs.%s.topology", graphID))

	if err != nil {
		return program.GraphModule{}, err
	}

	module.Topology = topology

	manifestPath, err := yamlString(body["manifest"], fmt.Sprintf("graphs.%s.manifest", graphID))

	if err != nil {
		return program.GraphModule{}, err
	}

	module.Manifest = manifestPath

	weightAsset, err := yamlString(body["weight_asset"], fmt.Sprintf("graphs.%s.weight_asset", graphID))

	if err != nil {
		return program.GraphModule{}, err
	}

	module.WeightAsset = weightAsset

	for key, value := range body {
		if key == "topology" || key == "manifest" || key == "weight_asset" {
			continue
		}

		module.Config[key] = value
	}

	return module, nil
}

func sortedKeys(input map[string]any) []string {
	keys := make([]string, 0, len(input))

	for key := range input {
		keys = append(keys, key)
	}

	sort.Strings(keys)

	return keys
}
