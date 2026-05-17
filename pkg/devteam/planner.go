package devteam

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	devcfg "github.com/theapemachine/caramba/pkg/config"
)

const plannerMaxIterations = 10

// plannerTools is the tool set for the Planner agent. It is intentionally
// read-only with respect to the filesystem — the Planner shapes context and
// produces subtasks, it never writes code.
var plannerTools = []ToolDefinition{
	{
		Name:        "search_code",
		Description: "Search the repository for a pattern (regex). Returns file:line:text hits.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"pattern":     map[string]any{"type": "string"},
				"max_results": map[string]any{"type": "integer"},
			},
			"required": []string{"pattern"},
		},
	},
	{
		Name:        "view_file",
		Description: "Read a file, optionally restricted to a line range.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path":      map[string]any{"type": "string"},
				"from_line": map[string]any{"type": "integer"},
				"to_line":   map[string]any{"type": "integer"},
			},
			"required": []string{"path"},
		},
	},
	{
		Name: "create_subtask",
		Description: `Create one subtask for this card. Call once per distinct unit of work.
Each subtask will be executed by an independent developer agent in its own
sandbox, so subtasks must be self-contained and not assume shared state.
Provide files_in_scope as the list of files the developer will need to touch.
Provide key_symbols as the function/type names most relevant to this subtask.
Provide sibling_notes as a map of other subtask titles to a one-line warning
about potential conflicts (e.g. "also modifies pkg/foo/bar.go").`,
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"title":       map[string]any{"type": "string"},
				"description": map[string]any{"type": "string", "description": "Detailed implementation guidance for the developer agent"},
				"files_in_scope": map[string]any{
					"type":  "array",
					"items": map[string]any{"type": "string"},
				},
				"key_symbols": map[string]any{
					"type":  "array",
					"items": map[string]any{"type": "string"},
				},
				"sibling_notes": map[string]any{
					"type": "object",
					"additionalProperties": map[string]any{
						"type": "string",
					},
				},
			},
			"required": []string{"title", "description", "files_in_scope"},
		},
	},
	{
		Name:        "done",
		Description: "Signal that all subtasks have been created.",
		Parameters: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	},
}

// ─────────────────────────────────────────────────────────────────────────────

/*
PlannerResult is the output of a Planner.Plan call: the list of subtask
definitions to be persisted to the database.
*/
type PlannerResult struct {
	Subtasks []SubtaskDraft
}

/*
SubtaskDraft is a subtask as produced by the Planner before it has been
assigned an ID or a developer agent.
*/
type SubtaskDraft struct {
	Title        string
	Description  string
	FilesInScope []string
	KeySymbols   []string
	SiblingNotes map[string]string
}

// ─────────────────────────────────────────────────────────────────────────────

/*
Planner is the first agent to act on a card. It uses read-only editor tools
to explore the repository and emits a set of focused, self-contained subtasks
that the orchestrator then fans out to individual developer agents.

The Planner is the context shaper: it narrows the blast radius per subtask,
identifies key symbols and files, and annotates potential inter-subtask
conflicts so downstream developers do not collide blindly.
*/
type Planner struct {
	ctx    context.Context
	llm    Provider
	editor *VirtualEditor
}

/*
NewPlanner constructs a Planner. It shares the VirtualEditor with the
orchestrator so its reads are visible in the read-set (no writes needed, but
the read-gate bookkeeping is useful for the editor's Search/View consistency).
*/
func NewPlanner(
	ctx context.Context,
	cfg devcfg.ProviderConfig,
	editor *VirtualEditor,
) *Planner {
	return &Planner{
		ctx:    ctx,
		llm:    NewProvider(cfg),
		editor: editor,
	}
}

/*
Plan runs the Planner agent loop. It receives the card title, description, and
the pre-extracted blast-radius markdown, then iterates with the LLM until the
agent calls done(), returning the collected subtask drafts.
*/
func (planner *Planner) Plan(
	title, description, blastContext string,
) (*PlannerResult, error) {
	system := fmt.Sprintf(`You are a senior software architect acting as a planning agent.

Your job is to decompose a feature request into a set of focused, self-contained
subtasks that can each be implemented independently by a separate developer agent.

Rules:
- Each subtask must be implementable without knowledge of what the other subtasks
  are doing at the code level (they run in separate containers). If two subtasks
  touch the same file, document that clearly in sibling_notes.
- Each subtask description must include enough implementation detail that a
  developer with no other context can execute it correctly. Include: which
  functions/types to create or modify, the expected interface/signature, and
  which tests to write.
- Use search_code and view_file to ground your decomposition in the actual
  codebase. Do not guess at file paths or function signatures.
- Aim for 2–6 subtasks. A single trivial card may have 1. A complex feature
  should not exceed 8.
- When you have created all subtasks, call done().

%s`, blastContext)

	userContent := fmt.Sprintf("Feature card: %s\n\n%s", title, description)

	history := []ChatMessage{
		{Role: "user", Content: userContent},
	}

	result := &PlannerResult{}

	for range plannerMaxIterations {
		started := time.Now()
		resp, err := planner.llm.Chat(planner.ctx, ChatRequest{
			System:    system,
			Messages:  history,
			Tools:     plannerTools,
			MaxTokens: 8192,
		})

		if err != nil {
			return nil, fmt.Errorf("planner: %w", err)
		}

		publishChatUsage("planner", started, resp)

		history = append(history, ChatMessage{
			Role:      "assistant",
			Content:   resp.Content,
			ToolCalls: resp.ToolCalls,
		})

		if len(resp.ToolCalls) == 0 {
			break
		}

		toolResults, done, err := planner.handleToolCalls(resp.ToolCalls, result)

		if err != nil {
			return nil, err
		}

		if done {
			break
		}

		history = append(history, toolResults...)
	}

	return result, nil
}

func (planner *Planner) handleToolCalls(
	calls []ToolCall, result *PlannerResult,
) ([]ChatMessage, bool, error) {
	messages := make([]ChatMessage, 0, len(calls))

	for _, call := range calls {
		output, done, err := planner.dispatchTool(call, result)

		if done {
			return nil, true, nil
		}

		if err != nil {
			output = "ERROR: " + err.Error()
		}

		messages = append(messages, ChatMessage{
			Role:       "tool",
			Content:    output,
			ToolCallID: call.ID,
		})
	}

	return messages, false, nil
}

func (planner *Planner) dispatchTool(
	call ToolCall, result *PlannerResult,
) (string, bool, error) {
	switch call.Name {
	case "search_code":
		pattern, _ := call.Input["pattern"].(string)
		maxResults := 40

		if v, ok := call.Input["max_results"].(float64); ok {
			maxResults = int(v)
		}

		hits, err := planner.editor.Search(pattern, maxResults)

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

		content, err := planner.editor.View(path, fromLine, toLine)

		return content, false, err

	case "create_subtask":
		draft, err := parseSubtaskDraft(call.Input)

		if err != nil {
			return "", false, fmt.Errorf("planner: create_subtask: %w", err)
		}

		result.Subtasks = append(result.Subtasks, draft)

		return fmt.Sprintf("subtask %q registered (total: %d)", draft.Title, len(result.Subtasks)), false, nil

	case "done":
		return "", true, nil

	default:
		return "", false, fmt.Errorf("unknown tool %q", call.Name)
	}
}

func parseSubtaskDraft(input map[string]any) (SubtaskDraft, error) {
	title, _ := input["title"].(string)
	description, _ := input["description"].(string)

	if title == "" {
		return SubtaskDraft{}, fmt.Errorf("title is required")
	}

	draft := SubtaskDraft{
		Title:        title,
		Description:  description,
		FilesInScope: toStringSlice(input["files_in_scope"]),
		KeySymbols:   toStringSlice(input["key_symbols"]),
		SiblingNotes: make(map[string]string),
	}

	if raw, ok := input["sibling_notes"]; ok {
		data, _ := json.Marshal(raw)
		_ = json.Unmarshal(data, &draft.SiblingNotes)
	}

	return draft, nil
}
