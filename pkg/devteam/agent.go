package devteam

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	devcfg "github.com/theapemachine/caramba/pkg/config"
)

const maxIterations = 10

/*
ReviewVerdict is the outcome returned by a Reviewer pass.
*/
type ReviewVerdict struct {
	Pass     bool
	Feedback string
}

var sandboxTools = []ToolDefinition{
	{
		Name:        "shell",
		Description: "Run a shell command in /workspace inside the container. Returns stdout+stderr.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"command": map[string]any{"type": "string", "description": "Shell command to execute"},
			},
			"required": []string{"command"},
		},
	},
	{
		Name:        "read_file",
		Description: "Read a file from /workspace by relative path.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{"type": "string"},
			},
			"required": []string{"path"},
		},
	},
	{
		Name:        "write_file",
		Description: "Create or overwrite a file in /workspace.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path":    map[string]any{"type": "string"},
				"content": map[string]any{"type": "string"},
			},
			"required": []string{"path", "content"},
		},
	},
	{
		Name:        "done",
		Description: "Signal that the feature implementation is complete.",
		Parameters: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	},
}

// ────────────────────────────────────────────────────────────────────────────

/*
Developer is an AI agent that reads a feature request, explores the repo inside
a Sandbox, writes code, and runs tests — iterating until it believes the feature
is complete or the maximum iteration count is reached.
*/
type Developer struct {
	ctx     context.Context
	llm     Provider
	sandbox *Sandbox
}

/*
NewDeveloper constructs a Developer bound to the given Sandbox and provider.
*/
func NewDeveloper(ctx context.Context, cfg devcfg.ProviderConfig, sandbox *Sandbox) *Developer {
	return &Developer{
		ctx:     ctx,
		llm:     NewProvider(cfg),
		sandbox: sandbox,
	}
}

/*
Implement drives the agentic loop: the Developer receives the card title and
description plus optional reviewer feedback and attempts to implement the
feature by calling shell tools inside the Sandbox.
*/
func (developer *Developer) Implement(title, description, feedback string) error {
	system := `You are a senior Go software engineer working inside a git repository.
You implement features by calling the provided shell tools.
Always write tests alongside code. Follow the repo's coding standards.
When you believe the feature is complete and tests pass, call done().`

	userContent := fmt.Sprintf("Feature: %s\n\n%s", title, description)

	if feedback != "" {
		userContent += fmt.Sprintf("\n\nReviewer feedback:\n%s", feedback)
	}

	history := []ChatMessage{
		{Role: "user", Content: userContent},
	}

	for range maxIterations {
		resp, err := developer.llm.Chat(developer.ctx, ChatRequest{
			System:    system,
			Messages:  history,
			Tools:     sandboxTools,
			MaxTokens: 8192,
		})

		if err != nil {
			return fmt.Errorf("developer: %w", err)
		}

		history = append(history, ChatMessage{
			Role:      "assistant",
			Content:   resp.Content,
			ToolCalls: resp.ToolCalls,
		})

		if len(resp.ToolCalls) == 0 {
			return nil
		}

		toolResults, done, err := developer.handleToolCalls(resp.ToolCalls)

		if err != nil {
			return err
		}

		if done {
			return nil
		}

		history = append(history, toolResults...)
	}

	return fmt.Errorf("developer: exceeded %d iterations without completing", maxIterations)
}

func (developer *Developer) handleToolCalls(
	calls []ToolCall,
) ([]ChatMessage, bool, error) {
	results := make([]ChatMessage, 0, len(calls))

	for _, call := range calls {
		var output string
		var callErr error

		switch call.Name {
		case "shell":
			cmd, _ := call.Input["command"].(string)
			output, callErr = developer.sandbox.Exec(cmd)

		case "read_file":
			path, _ := call.Input["path"].(string)
			output, callErr = developer.sandbox.ReadFile(path)

		case "write_file":
			path, _ := call.Input["path"].(string)
			content, _ := call.Input["content"].(string)
			callErr = developer.sandbox.WriteFile(path, content)
			output = "ok"

		case "done":
			return nil, true, nil

		default:
			callErr = fmt.Errorf("unknown tool %q", call.Name)
		}

		if callErr != nil {
			output = "ERROR: " + callErr.Error() + "\n" + output
		}

		results = append(results, ChatMessage{
			Role:       "tool",
			Content:    output,
			ToolCallID: call.ID,
		})
	}

	return results, false, nil
}

// ────────────────────────────────────────────────────────────────────────────

/*
Reviewer is an AI agent that inspects the changes made by the Developer inside
the Sandbox and decides whether they meet quality standards.
*/
type Reviewer struct {
	ctx context.Context
	llm Provider
}

/*
NewReviewer constructs a Reviewer.
*/
func NewReviewer(ctx context.Context, cfg devcfg.ProviderConfig) *Reviewer {
	return &Reviewer{
		ctx: ctx,
		llm: NewProvider(cfg),
	}
}

/*
Review inspects the git diff and test output inside the sandbox and returns a
ReviewVerdict. Pass=true means the PR can be opened; Pass=false means the
developer should iterate with the provided Feedback.
*/
func (reviewer *Reviewer) Review(sandbox *Sandbox, title, description string) (ReviewVerdict, error) {
	diff, err := sandbox.Exec(`git -C /workspace diff HEAD~1 HEAD 2>/dev/null || git -C /workspace diff --cached`)

	if err != nil {
		diff = "(could not read diff: " + err.Error() + ")"
	}

	testOutput, _ := sandbox.Exec(`cd /workspace && go test ./... 2>&1 | tail -60`)

	system := `You are a senior Go code reviewer.
Evaluate the provided git diff and test output for a feature implementation.
Reply ONLY with a JSON object: {"pass": true/false, "feedback": "..."}
Pass if the implementation is correct, has tests, and tests pass.
Fail with actionable feedback otherwise.`

	userContent := fmt.Sprintf(
		"Feature: %s\n\n%s\n\n--- git diff ---\n%s\n\n--- test output ---\n%s",
		title, description, truncate(diff, 6000), truncate(testOutput, 2000),
	)

	resp, err := reviewer.llm.Chat(reviewer.ctx, ChatRequest{
		System:    system,
		Messages:  []ChatMessage{{Role: "user", Content: userContent}},
		MaxTokens: 1024,
	})

	if err != nil {
		return ReviewVerdict{}, fmt.Errorf("reviewer: %w", err)
	}

	raw := strings.TrimSpace(resp.Content)

	var verdict struct {
		Pass     bool   `json:"pass"`
		Feedback string `json:"feedback"`
	}

	if err := json.Unmarshal([]byte(raw), &verdict); err != nil {
		return ReviewVerdict{Pass: false, Feedback: raw}, nil
	}

	return ReviewVerdict{Pass: verdict.Pass, Feedback: verdict.Feedback}, nil
}
