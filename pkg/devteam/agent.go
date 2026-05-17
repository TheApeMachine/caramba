package devteam

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/theapemachine/caramba/pkg/asset"
	devcfg "github.com/theapemachine/caramba/pkg/config"
	"gopkg.in/yaml.v3"
)

const (
	developerMaxIterations = 10
	agentAssetPath         = "devteam/agent.yml"
)

var agentAssets = loadAgentAssets()

var editorTools = agentAssets.Developer.Tools

/*
ReviewVerdict is the outcome returned by a Reviewer pass.
*/
type ReviewVerdict struct {
	Pass     bool
	Feedback string
}

/*
AgentAssets holds the embedded developer-team role prompts and tool schemas.
*/
type AgentAssets struct {
	Developer AgentRoleAssets `yaml:"developer"`
	Reviewer  AgentRoleAssets `yaml:"reviewer"`
}

/*
AgentRoleAssets holds the prompt and tool schemas for one LLM role.
*/
type AgentRoleAssets struct {
	SystemPrompt string           `yaml:"system_prompt"`
	Tools        []ToolDefinition `yaml:"tools"`
}

func loadAgentAssets() AgentAssets {
	data, err := asset.ReadFile(agentAssetPath)

	if err != nil {
		panic(fmt.Sprintf("devteam agent asset: %v", err))
	}

	var assets AgentAssets

	if err := yaml.Unmarshal(data, &assets); err != nil {
		panic(fmt.Sprintf("devteam agent asset: %v", err))
	}

	if err := assets.Validate(); err != nil {
		panic(fmt.Sprintf("devteam agent asset: %v", err))
	}

	return assets
}

/*
Validate verifies that embedded role prompts and tool schemas are usable.
*/
func (assets AgentAssets) Validate() error {
	if strings.TrimSpace(assets.Developer.SystemPrompt) == "" {
		return fmt.Errorf("developer system_prompt is required")
	}

	if !strings.Contains(assets.Developer.SystemPrompt, "{{blast_context}}") {
		return fmt.Errorf("developer system_prompt must include {{blast_context}}")
	}

	if strings.TrimSpace(assets.Reviewer.SystemPrompt) == "" {
		return fmt.Errorf("reviewer system_prompt is required")
	}

	return validateToolDefinitions("developer", assets.Developer.Tools)
}

func validateToolDefinitions(role string, tools []ToolDefinition) error {
	if len(tools) == 0 {
		return fmt.Errorf("%s tools are required", role)
	}

	seen := make(map[string]struct{}, len(tools))

	for toolIndex, toolDefinition := range tools {
		name := strings.TrimSpace(toolDefinition.Name)

		if name == "" {
			return fmt.Errorf("%s tool %d name is required", role, toolIndex)
		}

		if _, exists := seen[name]; exists {
			return fmt.Errorf("%s tool %q is duplicated", role, name)
		}

		if strings.TrimSpace(toolDefinition.Description) == "" {
			return fmt.Errorf("%s tool %q description is required", role, name)
		}

		if len(toolDefinition.Parameters) == 0 {
			return fmt.Errorf("%s tool %q parameters are required", role, name)
		}

		seen[name] = struct{}{}
	}

	return nil
}

func renderAgentPrompt(prompt string, values map[string]string) string {
	rendered := prompt

	for placeholder, value := range values {
		rendered = strings.ReplaceAll(rendered, "{{"+placeholder+"}}", value)
	}

	return strings.TrimSpace(rendered)
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
	system := developer.SystemPrompt(blastContext)

	userContent := fmt.Sprintf("Feature: %s\n\n%s", title, description)

	if feedback != "" {
		userContent += fmt.Sprintf("\n\nReviewer feedback from previous iteration:\n%s", feedback)
	}

	history := []ChatMessage{
		{Role: "user", Content: userContent},
	}

	for range developerMaxIterations {
		started := time.Now()
		resp, err := developer.llm.Chat(developer.ctx, ChatRequest{
			System:    system,
			Messages:  history,
			Tools:     editorTools,
			MaxTokens: 8192,
		})

		if err != nil {
			return fmt.Errorf("developer: %w", err)
		}

		publishChatUsage("developer", started, resp)

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

	return fmt.Errorf(
		"developer: exceeded %d iterations without completing",
		developerMaxIterations,
	)
}

/*
SystemPrompt renders the embedded developer prompt with task-specific context.
*/
func (developer *Developer) SystemPrompt(blastContext string) string {
	return renderAgentPrompt(agentAssets.Developer.SystemPrompt, map[string]string{
		"blast_context": blastContext,
	})
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
