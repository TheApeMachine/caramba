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

// editorTools is the canonical tool set exposed to the developer agent.
// search and view are always safe; edit/create require prior read and a lock.
var editorTools = []ToolDefinition{
	{
		Name:        "search_code",
		Description: "Search the repository for a pattern (regex). Returns file:line:text hits.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"pattern":     map[string]any{"type": "string", "description": "Grep -E regex"},
				"max_results": map[string]any{"type": "integer", "description": "Max hits (default 40)"},
			},
			"required": []string{"pattern"},
		},
	},
	{
		Name:        "view_file",
		Description: "Read a file. Optionally restrict to a line range. Always call this before edit_file.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path":      map[string]any{"type": "string"},
				"from_line": map[string]any{"type": "integer", "description": "1-based start line (0 = beginning)"},
				"to_line":   map[string]any{"type": "integer", "description": "1-based end line (0 = end)"},
			},
			"required": []string{"path"},
		},
	},
	{
		Name: "edit_file",
		Description: `Replace a contiguous block of lines in an existing file.
You MUST have called view_file on this path first.
Provide old_lines exactly as they appear in the file (trailing spaces ignored).
new_lines replaces that block verbatim.`,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path":         map[string]any{"type": "string"},
				"old_lines":    map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
				"new_lines":    map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
				"claim_intent": map[string]any{"type": "string", "description": "One-line description of what you are changing and why"},
			},
			"required": []string{"path", "old_lines", "new_lines", "claim_intent"},
		},
	},
	{
		Name:        "create_file",
		Description: "Create a new file. Do not use to overwrite existing files — use edit_file for that.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path":         map[string]any{"type": "string"},
				"content":      map[string]any{"type": "string"},
				"claim_intent": map[string]any{"type": "string"},
			},
			"required": []string{"path", "content", "claim_intent"},
		},
	},
	{
		Name:        "run_shell",
		Description: "Run a shell command in /workspace (build, test, lint). Avoid using this to read or write files.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"command": map[string]any{"type": "string"},
			},
			"required": []string{"command"},
		},
	},
	{
		Name: "sub_agent",
		Description: `Deploy one or more short-lived read-only sub-agents in parallel.
Each sub-agent receives its own system prompt and user prompt, runs in a separate
context window, and returns a concise result. Use this for isolated repository
research, code reading, or parallel analysis. Sub-agents cannot edit files or run
shell commands.`,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"tasks": map[string]any{
					"type": "array",
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"name":          map[string]any{"type": "string", "description": "Short stable name for this sub-agent task"},
							"system_prompt": map[string]any{"type": "string", "description": "Persona and constraints for the sub-agent"},
							"user_prompt":   map[string]any{"type": "string", "description": "Specific task the sub-agent should perform"},
						},
						"required": []string{"name", "system_prompt", "user_prompt"},
					},
				},
			},
			"required": []string{"tasks"},
		},
	},
	{
		Name: "done",
		Description: `Signal that the feature implementation is complete.
Before calling this you MUST:
1. Have written at least one test file covering the new/changed behaviour.
2. Have called run_shell with "go test ./..." (or a scoped equivalent) and
   confirmed the output shows no failures.
If either condition is unmet the tool will reject the signal and tell you what
is missing.`,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"test_file":    map[string]any{"type": "string", "description": "Path to the test file you wrote or extended"},
				"test_command": map[string]any{"type": "string", "description": "The exact go test command you ran"},
				"test_output":  map[string]any{"type": "string", "description": "Last output from go test (must show PASS or ok lines, no FAIL)"},
			},
			"required": []string{"test_file", "test_command", "test_output"},
		},
	},
}

// ─────────────────────────────────────────────────────────────────────────────

/*
Developer is an AI agent that reads a feature request, explores the repo inside
a Sandbox via a VirtualEditor, writes targeted changes, and runs tests —
iterating until it believes the feature is complete or the iteration cap is hit.
*/
type Developer struct {
	ctx    context.Context
	cfg    devcfg.ProviderConfig
	llm    Provider
	editor *VirtualEditor
	subs   *SubAgentPool
}

/*
NewDeveloper constructs a Developer bound to the given VirtualEditor.
*/
func NewDeveloper(
	ctx context.Context,
	cfg devcfg.ProviderConfig,
	editor *VirtualEditor,
) *Developer {
	return &Developer{
		ctx:    ctx,
		cfg:    cfg,
		llm:    NewProvider(cfg),
		editor: editor,
		subs:   NewSubAgentPool(ctx, cfg, editor),
	}
}

/*
Implement drives the agentic loop. blastContext is the pre-extracted blast
radius markdown injected into the system prompt. feedback is reviewer feedback
from the previous iteration (empty on first call).
*/
func (developer *Developer) Implement(
	title, description, blastContext, feedback string,
) error {
	system := fmt.Sprintf(`You are a senior Go software engineer working inside a git repository at /workspace.
You implement features using the provided tools. Rules:
- Always call view_file before edit_file on the same path.
- Prefer edit_file over create_file for existing files.
- Write tests alongside implementation code.
- Follow the repository's coding conventions (see CLAUDE.md if present).
- When all tests pass and the feature is complete, call done().

%s`, blastContext)

	userContent := fmt.Sprintf("Feature: %s\n\n%s", title, description)

	if feedback != "" {
		userContent += fmt.Sprintf("\n\nReviewer feedback from previous iteration:\n%s", feedback)
	}

	history := []ChatMessage{
		{Role: "user", Content: userContent},
	}

	for range maxIterations {
		resp, err := developer.llm.Chat(developer.ctx, ChatRequest{
			System:    system,
			Messages:  history,
			Tools:     editorTools,
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
		output, done, callErr := developer.dispatchTool(call)

		if done {
			return nil, true, nil
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

func (developer *Developer) dispatchTool(call ToolCall) (string, bool, error) {
	switch call.Name {
	case "search_code":
		pattern, _ := call.Input["pattern"].(string)
		maxResults := 40

		if v, ok := call.Input["max_results"].(float64); ok {
			maxResults = int(v)
		}

		hits, err := developer.editor.Search(pattern, maxResults)

		if err != nil {
			return "", false, err
		}

		var sb strings.Builder

		for _, hit := range hits {
			fmt.Fprintf(&sb, "%s:%d:%s\n", hit.Path, hit.Line, hit.Text)
		}

		return sb.String(), false, nil

	case "view_file":
		path, _ := call.Input["path"].(string)
		var fromLine, toLine uint32

		if v, ok := call.Input["from_line"].(float64); ok {
			fromLine = uint32(v)
		}

		if v, ok := call.Input["to_line"].(float64); ok {
			toLine = uint32(v)
		}

		content, err := developer.editor.View(path, fromLine, toLine)

		return content, false, err

	case "edit_file":
		path, _ := call.Input["path"].(string)
		intent, _ := call.Input["claim_intent"].(string)

		oldLines := toStringSlice(call.Input["old_lines"])
		newLines := toStringSlice(call.Input["new_lines"])

		err := developer.editor.Edit(EditRequest{
			Path:        path,
			OldLines:    oldLines,
			NewLines:    newLines,
			ClaimIntent: intent,
		})

		if err != nil {
			return "", false, err
		}

		return "ok", false, nil

	case "create_file":
		path, _ := call.Input["path"].(string)
		content, _ := call.Input["content"].(string)
		intent, _ := call.Input["claim_intent"].(string)

		err := developer.editor.Create(path, content, intent)

		return "ok", false, err

	case "run_shell":
		cmd, _ := call.Input["command"].(string)
		out, err := developer.editor.sandbox.Exec(cmd)

		return out, false, err

	case "sub_agent":
		return developer.subs.Dispatch(call.Input)

	case "done":
		rejection := developer.checkDoneGate(call.Input)

		if rejection != "" {
			return rejection, false, nil
		}

		return "", true, nil

	default:
		return "", false, fmt.Errorf("unknown tool %q", call.Name)
	}
}

/*
checkDoneGate verifies that the agent has satisfied the test contract before
accepting a done() signal. It returns a non-empty rejection message when any
condition is unmet, which is fed back to the LLM as a tool result so it knows
exactly what to fix.
*/
func (developer *Developer) checkDoneGate(input map[string]any) string {
	testFile, _ := input["test_file"].(string)
	testOutput, _ := input["test_output"].(string)

	if testFile == "" {
		return "REJECTED: test_file is required. Write at least one test file before calling done()."
	}

	if _, err := developer.editor.sandbox.ReadFile(testFile); err != nil {
		return fmt.Sprintf(
			"REJECTED: test file %q does not exist in the workspace. Create it before calling done().",
			testFile,
		)
	}

	lower := strings.ToLower(testOutput)

	if strings.Contains(lower, "fail") || !strings.Contains(lower, "ok") {
		return fmt.Sprintf(
			"REJECTED: test output does not confirm a clean pass. Run go test and ensure all tests pass before calling done().\nOutput seen:\n%s",
			truncate(testOutput, 800),
		)
	}

	return ""
}

func toStringSlice(v any) []string {
	raw, ok := v.([]any)

	if !ok {
		return nil
	}

	out := make([]string, 0, len(raw))

	for _, item := range raw {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}

	return out
}

// ─────────────────────────────────────────────────────────────────────────────

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
ReviewVerdict.
*/
func (reviewer *Reviewer) Review(
	sandbox *Sandbox, title, description string,
) (ReviewVerdict, error) {
	diff, err := sandbox.Exec(
		`git -C /workspace diff HEAD~1 HEAD 2>/dev/null || git -C /workspace diff --cached`,
	)

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
