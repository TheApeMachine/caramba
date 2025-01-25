package ai

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"time"
)

/*
Interpreter is an object that extracts and interprets commands from unstructured text.
It maps any commands to handler methods.
*/
type Interpreter struct {
	text     string
	commands []map[string]func()
}

/*
NewInterpreter creates a new Interpreter.
*/
func NewInterpreter(text string) *Interpreter {
	return &Interpreter{
		text:     text,
		commands: make([]map[string]func(), 0),
	}
}

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

func (interpreter *Interpreter) Execute() {
	for _, commandMap := range interpreter.commands {
		for _, action := range commandMap {
			action()
		}
	}
}

func (interpreter *Interpreter) Interpret() *Interpreter {
	regexpattern := regexp.MustCompile(`<(\w+)(?:\s+(?:(\w+)\s*=\s*"([^"]*)")\s*)*>`)
	matches := regexpattern.FindAllStringSubmatch(interpreter.text, -1)

	for _, match := range matches {
		command := match[1]
		args := interpreter.getArguments(match[2:])

		action := func() {
			switch command {
			case "STORE":
				interpreter.handleStore(args)
			case "IGNORE":
				reason := args["reason"]
				if reason == "" {
					reason = "No reason provided"
				}
				fmt.Printf("Ignoring response: %s\n", reason)
			default:
				fmt.Printf("Command %s with arguments: %v\n", command, args)
			}
		}

		interpreter.commands = append(interpreter.commands, map[string]func(){
			command: action,
		})
	}

	return interpreter
}

func (interpreter *Interpreter) handleStore(args map[string]string) {
	// Create artifacts directory if it doesn't exist
	artifactsDir := "artifacts"
	if err := os.MkdirAll(artifactsDir, 0755); err != nil {
		fmt.Printf("Error creating artifacts directory: %v\n", err)
		return
	}

	// Generate timestamp-based filename
	timestamp := time.Now().Format("20060102-150405")
	filename := filepath.Join(artifactsDir, fmt.Sprintf("finetuning-%s.jsonl", timestamp))

	// Create or open the file
	file, err := os.Create(filename)
	if err != nil {
		fmt.Printf("Error creating file: %v\n", err)
		return
	}
	defer file.Close()

	// Parse and write artifacts
	if artifactsStr, ok := args["artifacts"]; ok {
		var artifacts []FinetuningArtifact
		if err := json.Unmarshal([]byte(artifactsStr), &artifacts); err != nil {
			fmt.Printf("Error parsing artifacts: %v\n", err)
			return
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
}

func (interpreter *Interpreter) getArguments(match []string) map[string]string {
	arguments := make(map[string]string)

	// Process pairs of key-value matches
	for i := 0; i < len(match); i += 2 {
		if match[i] != "" && i+1 < len(match) {
			arguments[match[i]] = match[i+1]
		}
	}

	return arguments
}
