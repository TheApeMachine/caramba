package train

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
CheckpointSave persists the incoming parameter vector to disk.
Input: data[0] = params. Output: passthrough of params.

Config keys:

	dir    — directory to write into (default "checkpoints")
	prefix — filename prefix (default "ckpt")

The node is stateful: it increments a call counter used in the filename.
*/
type CheckpointSave struct {
}

func NewCheckpointSave(dir, prefix string) *CheckpointSave {
	return &CheckpointSave{}
}

func (checkpointSave *CheckpointSave) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.Err(); err != nil {
		return nil, err
	}

	params := stateDict.Params

	if len(params) == 0 && len(stateDict.Inputs) > 0 {
		params = stateDict.Inputs[0]
	}

	if len(params) == 0 {
		return nil, fmt.Errorf("train.checkpoint_save: params are required")
	}

	dir := stateDict.Cache

	if dir == "" {
		dir = "checkpoints"
	}

	prefix := stateDict.Name

	if prefix == "" {
		prefix = "ckpt"
	}

	stateDict.Count++

	err := os.MkdirAll(dir, 0o755)

	if err != nil {
		return nil, err
	}

	path := filepath.Join(dir, fmt.Sprintf("%s_%06d.json", prefix, stateDict.Count))
	raw, err := json.Marshal(params)

	if err != nil {
		return nil, err
	}

	if err := os.WriteFile(path, raw, 0o644); err != nil {
		return nil, err
	}

	stateDict.SetOperationOutput(append(stateDict.Out[:0], params...))

	return stateDict, nil
}

/*
CheckpointLoad reads a parameter vector from disk and emits it.
It takes no meaningful inputs; shape is ignored.

Config keys:

	path — file to load (required)
*/
type CheckpointLoad struct {
}

func NewCheckpointLoad(path string) *CheckpointLoad {
	return &CheckpointLoad{}
}

func (checkpointLoad *CheckpointLoad) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.Err(); err != nil {
		return nil, err
	}

	if stateDict.File == "" {
		return nil, fmt.Errorf("train.checkpoint_load: file is required")
	}

	raw, err := os.ReadFile(stateDict.File)

	if err != nil {
		return nil, err
	}

	var params []float64

	if err := json.Unmarshal(raw, &params); err != nil {
		return nil, err
	}

	stateDict.SetOperationOutput(params)

	return stateDict, nil
}
