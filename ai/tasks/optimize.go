package tasks

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/theapemachine/caramba/ai/drknow"
)

// FinetuningMessage represents a single message in the fine-tuning data
type FinetuningMessage struct {
	Role    string  `json:"role"`
	Content string  `json:"content"`
	Weight  float64 `json:"weight,omitempty"`
}

// FinetuningArtifact represents a complete training example
type FinetuningArtifact struct {
	Messages []FinetuningMessage `json:"messages"`
}

type Optimize struct {
}

func NewOptimize() *Optimize {
	return &Optimize{}
}

func (o *Optimize) Execute(
	ctx *drknow.Context,
	args map[string]any,
) Bridge {
	// Create artifacts directory if it doesn't exist
	artifactsDir := "artifacts"
	if err := os.MkdirAll(artifactsDir, 0755); err != nil {
		fmt.Printf("Error creating artifacts directory: %v\n", err)
	}

	// Generate timestamp-based filename
	timestamp := time.Now().Format("20060102-150405")
	filename := filepath.Join(artifactsDir, fmt.Sprintf("finetuning-%s.jsonl", timestamp))

	// Create or open the file
	file, err := os.Create(filename)
	if err != nil {
		fmt.Printf("Error creating file: %v\n", err)
	}
	defer file.Close()

	// Parse and write artifacts
	if artifactsStr, ok := args["artifacts"].(string); ok {
		var artifacts []FinetuningArtifact
		if err := json.Unmarshal([]byte(artifactsStr), &artifacts); err != nil {
			fmt.Printf("Error parsing artifacts: %v\n", err)
		}

		// Write each artifact as a separate line
		encoder := json.NewEncoder(file)
		for _, artifact := range artifacts {
			if err := encoder.Encode(artifact); err != nil {
				fmt.Printf("Error writing artifact: %v\n", err)
				continue
			}
		}

		fmt.Printf("Successfully wrote artifacts to %s\n", filename)
	} else {
		fmt.Println("No artifacts provided in the STORE command")
	}

	return nil
}
