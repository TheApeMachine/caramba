package train

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
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
	dir    string
	prefix string
	calls  int
}

func NewCheckpointSave(dir, prefix string) *CheckpointSave {
	if dir == "" {
		dir = "checkpoints"
	}

	if prefix == "" {
		prefix = "ckpt"
	}

	return &CheckpointSave{dir: dir, prefix: prefix}
}

func (cs *CheckpointSave) Forward(_ []int, data ...[]float64) []float64 {
	cs.calls++

	err := os.MkdirAll(cs.dir, 0o755)

	if err != nil {
		return data[0]
	}

	path := filepath.Join(cs.dir, fmt.Sprintf("%s_%06d.json", cs.prefix, cs.calls))
	raw, _ := json.Marshal(data[0])
	_ = os.WriteFile(path, raw, 0o644)

	return data[0]
}

/*
CheckpointLoad reads a parameter vector from disk and emits it.
It takes no meaningful inputs; shape is ignored.

Config keys:
  path — file to load (required)
*/
type CheckpointLoad struct {
	path   string
	params []float64
}

func NewCheckpointLoad(path string) *CheckpointLoad {
	return &CheckpointLoad{path: path}
}

func (cl *CheckpointLoad) Forward(_ []int, _ ...[]float64) []float64 {
	if cl.params != nil {
		return cl.params
	}

	raw, err := os.ReadFile(cl.path)

	if err != nil {
		return nil
	}

	var params []float64

	if json.Unmarshal(raw, &params) != nil {
		return nil
	}

	cl.params = params

	return cl.params
}
