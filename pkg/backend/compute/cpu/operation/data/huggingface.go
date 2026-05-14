package data

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

const datasetsBaseURL = "https://datasets-server.huggingface.co"

/*
HuggingFace streams rows from the HuggingFace Datasets Server REST API.
Each Forward call returns the next page of flattened float64 values.
The shape slice encodes [rows, cols] so downstream nodes can reshape.

Config keys:

	dataset  — e.g. "glue"
	config   — dataset config name, e.g. "sst2" (default "default")
	split    — "train", "validation", "test" (default "train")
	page     — rows per Forward call (default 100)
	field    — which row field to extract as float64 (default "label")
*/
type HuggingFace struct {
	ctx     context.Context
	cancel  context.CancelFunc
	client  *http.Client
	dataset string
	config  string
	split   string
	field   string
	page    int
	offset  int
	done    bool
}

/*
NewHuggingFace creates a HuggingFace data source node.
*/
func NewHuggingFace(
	dataset, datasetConfig, split, field string, page int,
) *HuggingFace {
	ctx, cancel := context.WithCancel(context.Background())

	return &HuggingFace{
		ctx:     ctx,
		cancel:  cancel,
		client:  &http.Client{},
		dataset: dataset,
		config:  datasetConfig,
		split:   split,
		field:   field,
		page:    page,
	}
}

/*
Forward fetches the next page of rows and returns them as a flat float64 slice.
shape[0] = actual rows returned, shape[1] = 1 (scalar field per row).
Returns nil when the dataset is exhausted.
*/
func (hf *HuggingFace) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.Err(); err != nil {
		return nil, err
	}

	if hf.done {
		stateDict.SetOperationOutput(nil)

		return stateDict, nil
	}

	rows, err := hf.fetchPage()

	if err != nil {
		hf.done = true

		return nil, err
	}

	if len(rows) == 0 {
		hf.done = true
		stateDict.SetOperationOutput(nil)

		return stateDict, nil
	}

	stateDict.WithShape([]int{len(rows), 1})
	stateDict.SetOperationOutput(rows)

	return stateDict, nil
}

/*
Reset repositions the stream to the beginning.
*/
func (hf *HuggingFace) Reset() {
	hf.offset = 0
	hf.done = false
}

/*
Close releases the underlying HTTP context.
*/
func (hf *HuggingFace) Close() {
	hf.cancel()
}

func (hf *HuggingFace) fetchPage() ([]float64, error) {
	endpoint := fmt.Sprintf(
		"%s/rows?dataset=%s&config=%s&split=%s&offset=%d&length=%d",
		datasetsBaseURL,
		url.QueryEscape(hf.dataset),
		url.QueryEscape(hf.config),
		url.QueryEscape(hf.split),
		hf.offset,
		hf.page,
	)

	req, err := http.NewRequestWithContext(hf.ctx, http.MethodGet, endpoint, nil)

	if err != nil {
		return nil, fmt.Errorf("data.huggingface: build request: %w", err)
	}

	resp, err := hf.client.Do(req)

	if err != nil {
		return nil, fmt.Errorf("data.huggingface: fetch: %w", err)
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)

		return nil, fmt.Errorf("data.huggingface: HTTP %d: %s", resp.StatusCode, body)
	}

	var payload struct {
		Rows []struct {
			Row map[string]any `json:"row"`
		} `json:"rows"`
	}

	err = json.NewDecoder(resp.Body).Decode(&payload)

	if err != nil {
		return nil, fmt.Errorf("data.huggingface: decode: %w", err)
	}

	out := make([]float64, 0, len(payload.Rows))

	for _, entry := range payload.Rows {
		out = append(out, extractFloat(entry.Row[hf.field]))
	}

	hf.offset += len(payload.Rows)

	if len(payload.Rows) < hf.page {
		hf.done = true
	}

	return out, nil
}

func extractFloat(v any) float64 {
	switch cast := v.(type) {
	case float64:
		return cast
	case int:
		return float64(cast)
	case bool:
		if cast {
			return 1
		}

		return 0
	default:
		return math.NaN()
	}
}
