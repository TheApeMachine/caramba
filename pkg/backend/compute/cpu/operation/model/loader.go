package model

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	"github.com/theapemachine/caramba/pkg/config"
	"github.com/theapemachine/caramba/pkg/hub"
)

/*
Loader fetches a model from HuggingFace or a local path and emits its
WeightMap encoded as a flat float64 slice (keys serialised as JSON in
the first element — a sentinel — so downstream nodes can deserialise).

In practice the Graph state carries []float64, but for the model loader
we encode the WeightMap as JSON bytes reinterpreted as float64 via a
shared convention: the output binding "model_weights" carries a single
float64 whose bit pattern is a pointer — we avoid that hack and instead
let the Loader satisfy operation.Operation while storing the WeightMap
internally for other model.* nodes to retrieve via the Registry.

Config keys:

	source  — HuggingFace repo ("org/name") or absolute local path
	file    — filename within the repo (default: "model.safetensors")
	cache   — local Hub cache directory (default: hub.cache_dir)
	revision — Git branch, tag, PR ref, or full commit hash (default: "main")
	repo_type — model, dataset, or space (default: "model")
*/
type Loader struct {
	source   string
	file     string
	cache    string
	revision string
	repoType string
	weights  WeightMap
}

/*
NewLoader creates a Loader node.
*/
func NewLoader(source, file, cache string) *Loader {
	if file == "" {
		file = "model.safetensors"
	}

	return &Loader{source: source, file: file, cache: cache}
}

/*
Forward loads the model on first call and emits a single float64 token
(the number of weight tensors) so the graph can sequence downstream ops.
The actual weights are held in Loader.weights and retrieved by downstream
model.* nodes via the shared WeightRegistry.
*/
func (loader *Loader) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.Err(); err != nil {
		return nil, err
	}

	source := stateDict.Source

	if source == "" {
		return nil, fmt.Errorf("model.loader: Source is required")
	}

	if weights, ok := globalRegistry.Get(source); ok {
		loader.weights = weights
		stateDict.SetOperationOutput([]float64{float64(len(weights))})

		return stateDict, nil
	}

	file := stateDict.File

	if file == "" {
		file = "model.safetensors"
	}

	cache := stateDict.Cache

	configured := &Loader{
		source:   source,
		file:     file,
		cache:    cache,
		revision: stateDict.Revision,
		repoType: stateDict.RepoType,
	}
	weights, err := configured.load()

	if err != nil {
		return nil, err
	}

	loader.weights = weights
	globalRegistry.store(source, weights)
	stateDict.SetOperationOutput([]float64{float64(len(weights))})

	return stateDict, nil
}

/*
Weights returns the loaded WeightMap directly, bypassing the graph state.
Used by surgery/graft/lora nodes that share the same source key.
*/
func (loader *Loader) Weights() WeightMap {
	return loader.weights
}

func (loader *Loader) load() (WeightMap, error) {
	localPath, err := loader.resolve()

	if err != nil {
		return nil, err
	}

	return Load(localPath)
}

func (loader *Loader) resolve() (string, error) {
	if filepath.IsAbs(loader.source) || strings.HasPrefix(loader.source, "./") {
		return filepath.Join(loader.source, loader.file), nil
	}

	location, err := hub.ParseLocator(loader.source)

	if err != nil {
		return "", err
	}

	if loader.revision != "" {
		location.Revision = loader.revision
	}

	if loader.repoType != "" {
		repoType, err := parseHubRepoType(loader.repoType)

		if err != nil {
			return "", err
		}

		location.RepoType = repoType
	}

	hubConfig := config.NewHubConfig()

	if loader.cache != "" {
		hubConfig.CacheDir = loader.cache
	}

	file, err := hub.NewClient(hubConfig).Download(
		context.Background(),
		hub.DownloadRequest{
			RepoID:   location.RepoID,
			RepoType: location.RepoType,
			Revision: location.Revision,
			Filename: loader.file,
		},
	)

	if err != nil {
		return "", fmt.Errorf("model.loader: hub download: %w", err)
	}

	return file.Path, nil
}

func parseHubRepoType(value string) (hub.RepoType, error) {
	switch hub.RepoType(strings.TrimSpace(value)) {
	case "", hub.ModelRepo:
		return hub.ModelRepo, nil
	case hub.DatasetRepo:
		return hub.DatasetRepo, nil
	case hub.SpaceRepo:
		return hub.SpaceRepo, nil
	default:
		return "", fmt.Errorf("model.loader: unsupported repo_type %q", value)
	}
}

/*
WeightRegistry is a process-level store so that surgery/graft/lora nodes
can retrieve the WeightMap loaded by a Loader node sharing the same source.
This avoids threading large tensors through the float64 graph state.
*/
type WeightRegistry struct {
	entries map[string]WeightMap
}

var globalRegistry = &WeightRegistry{entries: make(map[string]WeightMap)}

func (registry *WeightRegistry) store(source string, weights WeightMap) {
	registry.entries[source] = weights
}

/*
Get retrieves a WeightMap by source key.
*/
func (registry *WeightRegistry) Get(source string) (WeightMap, bool) {
	weights, ok := registry.entries[source]

	return weights, ok
}

/*
GlobalRegistry exposes the process-level weight store for use by other
model.* nodes. Using the same source string as the Loader is the contract.
*/
func GlobalRegistry() *WeightRegistry {
	return globalRegistry
}

/*
StoreForTest seeds the registry with a pre-built WeightMap, used in tests
to avoid disk or network access.
*/
func (registry *WeightRegistry) StoreForTest(source string, weights WeightMap) {
	registry.store(source, weights)
}

// jsonFloat64 is used to round-trip a JSON blob through a []float64 by
// encoding bytes as float64 values. We prefer the registry approach above
// and keep this only for the wire protocol if ever needed.
func jsonFloat64(v any) ([]float64, error) {
	b, err := json.Marshal(v)

	if err != nil {
		return nil, err
	}

	out := make([]float64, len(b))

	for idx, byte_ := range b {
		out[idx] = float64(byte_)
	}

	return out, nil
}

func float64JSON(data []float64, target any) error {
	b := make([]byte, len(data))

	for idx, f := range data {
		b[idx] = byte(f)
	}

	return json.Unmarshal(b, target)
}
