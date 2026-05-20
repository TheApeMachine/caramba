package devteam

import (
	"context"
	"fmt"
	"strings"
	"time"

	devcfg "github.com/theapemachine/caramba/pkg/config"
	"github.com/theapemachine/qpool"
)

const subAgentTimeout = 2 * time.Minute

const subAgentMaxTokens = 4096

const subAgentMaxIterations = 10

/*
SubAgentTask is a single short-lived delegation requested by the developer.
*/
type SubAgentTask struct {
	Name         string
	SystemPrompt string
	UserPrompt   string
}

/*
SubAgentResult is the ordered response returned to the developer agent.
*/
type SubAgentResult struct {
	Name   string
	Output string
	Error  error
}

/*
SubAgentMessage is a coordination event exchanged by sibling sub-agents.
*/
type SubAgentMessage struct {
	From    string
	Content string
	At      time.Time
}

/*
SubAgentPool dispatches read-only LLM workers through qpool so the developer can
parallelise focused analysis without sharing its active context window.
*/
type SubAgentPool struct {
	ctx         context.Context
	cfg         devcfg.ProviderConfig
	editor      readOnlyEditor
	queue       *qpool.Q
	newProvider func(devcfg.ProviderConfig) Provider
}

type subAgent struct {
	ctx          context.Context
	name         string
	provider     Provider
	editor       readOnlyEditor
	coordination *SubAgentCoordination
}

/*
SubAgentCoordination owns one qpool broadcast group for a single sub_agent call.
*/
type SubAgentCoordination struct {
	group  *qpool.BroadcastGroup
	inbox  map[string]chan *qpool.QValue
	replay map[string][]SubAgentMessage
}

type readOnlyEditor interface {
	Search(pattern string, maxResults int) ([]SearchResult, error)
	View(path string, fromLine, toLine uint32) (string, error)
}

/*
NewSubAgentPool constructs a bounded qpool-backed dispatcher for sub-agents.
*/
func NewSubAgentPool(
	ctx context.Context, cfg devcfg.ProviderConfig, editor *VirtualEditor,
) *SubAgentPool {
	config := qpool.NewConfig()
	config.Scaler = nil
	config.JobChannelCapacity = 16
	config.SchedulingTimeout = 10 * time.Second

	return &SubAgentPool{
		ctx:         ctx,
		cfg:         cfg,
		editor:      editor,
		queue:       qpool.NewQ(ctx, 4, 4, config),
		newProvider: NewProvider,
	}
}

/*
Dispatch validates the tool input, runs all requested sub-agents concurrently,
and renders their outputs in request order for the parent developer.
*/
func (pool *SubAgentPool) Dispatch(input map[string]any) (string, bool, error) {
	tasks, err := pool.Parse(input)

	if err != nil {
		return "", false, err
	}

	results := pool.Run(tasks)

	return pool.Format(results), false, nil
}

/*
Parse converts provider-decoded JSON tool input into typed sub-agent tasks.
*/
func (pool *SubAgentPool) Parse(input map[string]any) ([]SubAgentTask, error) {
	rawTasks, ok := input["tasks"].([]any)

	if !ok || len(rawTasks) == 0 {
		return nil, fmt.Errorf("sub_agent: tasks is required")
	}

	tasks := make([]SubAgentTask, 0, len(rawTasks))

	for index, rawTask := range rawTasks {
		item, ok := rawTask.(map[string]any)

		if !ok {
			return nil, fmt.Errorf("sub_agent: task %d must be an object", index)
		}

		task := SubAgentTask{
			Name:         strings.TrimSpace(stringField(item, "name")),
			SystemPrompt: strings.TrimSpace(stringField(item, "system_prompt")),
			UserPrompt:   strings.TrimSpace(stringField(item, "user_prompt")),
		}

		if task.Name == "" || task.SystemPrompt == "" || task.UserPrompt == "" {
			return nil, fmt.Errorf("sub_agent: task %d requires name, system_prompt, and user_prompt", index)
		}

		tasks = append(tasks, task)
	}

	return tasks, nil
}

/*
Run schedules every task on qpool and waits for completion in request order.
*/
func (pool *SubAgentPool) Run(tasks []SubAgentTask) []SubAgentResult {
	channels := make([]chan *qpool.QValue, len(tasks))
	coordination := pool.NewCoordination(tasks)

	defer coordination.Close()

	for index, task := range tasks {
		task := task
		jobID := fmt.Sprintf("sub-agent-%d-%s", index, task.Name)

		channels[index] = pool.queue.Schedule(
			jobID,
			func(ctx context.Context) (any, error) {
				return pool.Execute(ctx, task, coordination)
			},
			qpool.WithExecTimeout(subAgentTimeout),
			qpool.WithTTL(time.Minute),
		)
	}

	results := make([]SubAgentResult, len(tasks))

	for index, channel := range channels {
		value := <-channel
		results[index] = SubAgentResult{Name: tasks[index].Name}

		if value.Error != nil {
			results[index].Error = value.Error
			continue
		}

		output, ok := value.Value.(string)

		if !ok {
			results[index].Error = fmt.Errorf("sub_agent: unexpected result type %T", value.Value)
			continue
		}

		results[index].Output = output
	}

	return results
}

/*
Execute runs a read-only sub-agent conversation using the same provider config.
*/
func (pool *SubAgentPool) Execute(
	ctx context.Context, task SubAgentTask, coordination *SubAgentCoordination,
) (string, error) {
	agent := &subAgent{
		ctx:          ctx,
		name:         task.Name,
		provider:     pool.newProvider(pool.cfg),
		editor:       pool.editor,
		coordination: coordination,
	}

	return agent.Run(pool.SystemPrompt(task), task.UserPrompt)
}

/*
SystemPrompt wraps the caller-provided persona with non-negotiable boundaries.
*/
func (pool *SubAgentPool) SystemPrompt(task SubAgentTask) string {
	return fmt.Sprintf(`%s

You are a short-lived sub-agent for the developer agent.
Rules:
- Stay focused on the user prompt.
- Use only read-only tools when tools are available.
- Coordinate with sibling sub-agents through publish_finding, read_peer_findings, and list_peers.
- Publish useful discoveries before finalizing when they may affect another sub-agent.
- Do not edit files, run shell commands, or claim to have changed code.
- Return concise findings with file paths and line numbers when relevant.`, task.SystemPrompt)
}

/*
Format renders sub-agent results for the parent agent.
*/
func (pool *SubAgentPool) Format(results []SubAgentResult) string {
	var builder strings.Builder

	for _, result := range results {
		fmt.Fprintf(&builder, "## %s\n", result.Name)

		if result.Error != nil {
			fmt.Fprintf(&builder, "ERROR: %s\n\n", result.Error)
			continue
		}

		fmt.Fprintf(&builder, "%s\n\n", result.Output)
	}

	return strings.TrimSpace(builder.String())
}

func readOnlySubAgentTools() []ToolDefinition {
	return []ToolDefinition{
		editorTools[0],
		editorTools[1],
		{
			Name:        "publish_finding",
			Description: "Broadcast a concise finding to sibling sub-agents working in this same delegation.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"content": map[string]any{"type": "string", "description": "Finding, assumption, risk, or file reference to share"},
				},
				"required": []string{"content"},
			},
		},
		{
			Name:        "read_peer_findings",
			Description: "Read findings already broadcast by sibling sub-agents.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"max_messages": map[string]any{"type": "integer", "description": "Maximum messages to drain from the peer inbox"},
				},
			},
		},
		{
			Name:        "list_peers",
			Description: "List sibling sub-agents currently subscribed to the coordination room.",
			Parameters:  map[string]any{"type": "object", "properties": map[string]any{}},
		},
	}
}

/*
NewCoordination creates and primes a qpool broadcast room for one sub_agent call.
*/
func (pool *SubAgentPool) NewCoordination(tasks []SubAgentTask) *SubAgentCoordination {
	groupID := fmt.Sprintf("sub-agent-coordination-%d", time.Now().UnixNano())
	coordination := &SubAgentCoordination{
		group:  pool.queue.CreateBroadcastGroup(groupID, time.Minute),
		inbox:  make(map[string]chan *qpool.QValue, len(tasks)),
		replay: make(map[string][]SubAgentMessage, len(tasks)),
	}

	for _, task := range tasks {
		coordination.inbox[task.Name] = coordination.group.Subscribe(task.Name, 32)
	}

	return coordination
}

func (coordination *SubAgentCoordination) Publish(from, content string) string {
	message := SubAgentMessage{
		From:    from,
		Content: strings.TrimSpace(content),
		At:      time.Now(),
	}

	if message.Content == "" {
		return "ignored empty finding"
	}

	coordination.group.Send(qpool.NewQValue(message))

	for peer := range coordination.inbox {
		if peer == from {
			continue
		}

		coordination.replay[peer] = append(coordination.replay[peer], message)
	}

	return "published"
}

func (coordination *SubAgentCoordination) Read(peer string, maxMessages int) string {
	if maxMessages <= 0 {
		maxMessages = 10
	}

	messages := make([]SubAgentMessage, 0, maxMessages)
	messages = append(messages, coordination.takeReplay(peer, maxMessages)...)

	if len(messages) >= maxMessages {
		return formatSubAgentMessages(messages)
	}

	inbox := coordination.inbox[peer]

	for len(messages) < maxMessages {
		select {
		case value := <-inbox:
			message, ok := value.Value.(SubAgentMessage)

			if !ok || message.From == peer {
				continue
			}

			messages = append(messages, message)

		default:
			return formatSubAgentMessages(messages)
		}
	}

	return formatSubAgentMessages(messages)
}

func (coordination *SubAgentCoordination) takeReplay(peer string, maxMessages int) []SubAgentMessage {
	pending := coordination.replay[peer]

	if len(pending) == 0 {
		return nil
	}

	limit := maxMessages

	if len(pending) < limit {
		limit = len(pending)
	}

	messages := append([]SubAgentMessage(nil), pending[:limit]...)
	coordination.replay[peer] = pending[limit:]

	return messages
}

func (coordination *SubAgentCoordination) Peers() string {
	peers := coordination.group.SubscriberIDs()

	if len(peers) == 0 {
		return "no peers subscribed"
	}

	return strings.Join(peers, "\n")
}

func (coordination *SubAgentCoordination) Close() {
	coordination.group.Close()
}

func (agent *subAgent) Run(system, userPrompt string) (string, error) {
	history := []ChatMessage{{Role: "user", Content: userPrompt}}
	lastContent := ""

	for range subAgentMaxIterations {
		started := time.Now()
		response, err := agent.provider.Chat(agent.ctx, ChatRequest{
			System:    system,
			Messages:  history,
			Tools:     readOnlySubAgentTools(),
			MaxTokens: subAgentMaxTokens,
		})

		if err != nil {
			return "", err
		}

		publishChatUsage("sub_agent."+agent.name, started, response)

		lastContent = strings.TrimSpace(response.Content)
		history = append(history, ChatMessage{
			Role:      "assistant",
			Content:   response.Content,
			ToolCalls: response.ToolCalls,
		})

		if len(response.ToolCalls) == 0 {
			return lastContent, nil
		}

		toolResults, err := agent.HandleToolCalls(response.ToolCalls)

		if err != nil {
			return "", err
		}

		history = append(history, toolResults...)
	}

	return lastContent, nil
}

func (agent *subAgent) HandleToolCalls(calls []ToolCall) ([]ChatMessage, error) {
	results := make([]ChatMessage, 0, len(calls))

	for _, call := range calls {
		output, err := agent.DispatchTool(call)

		if err != nil {
			output = "ERROR: " + err.Error()
		}

		results = append(results, ChatMessage{
			Role:       "tool",
			Content:    output,
			ToolCallID: call.ID,
		})
	}

	return results, nil
}

func (agent *subAgent) DispatchTool(call ToolCall) (string, error) {
	switch call.Name {
	case "search_code":
		pattern, _ := call.Input["pattern"].(string)
		maxResults := 40

		if value, ok := call.Input["max_results"].(float64); ok {
			maxResults = int(value)
		}

		hits, err := agent.editor.Search(pattern, maxResults)

		if err != nil {
			return "", err
		}

		var builder strings.Builder

		for _, hit := range hits {
			fmt.Fprintf(&builder, "%s:%d:%s\n", hit.Path, hit.Line, hit.Text)
		}

		return builder.String(), nil

	case "view_file":
		path, _ := call.Input["path"].(string)
		var fromLine, toLine uint32

		if value, ok := call.Input["from_line"].(float64); ok {
			fromLine = uint32(value)
		}

		if value, ok := call.Input["to_line"].(float64); ok {
			toLine = uint32(value)
		}

		return agent.editor.View(path, fromLine, toLine)

	case "publish_finding":
		content, _ := call.Input["content"].(string)

		return agent.coordination.Publish(agent.name, content), nil

	case "read_peer_findings":
		maxMessages := 10

		if value, ok := call.Input["max_messages"].(float64); ok {
			maxMessages = int(value)
		}

		return agent.coordination.Read(agent.name, maxMessages), nil

	case "list_peers":
		return agent.coordination.Peers(), nil

	default:
		return "", fmt.Errorf("sub_agent: tool %q is not available", call.Name)
	}
}

func formatSubAgentMessages(messages []SubAgentMessage) string {
	if len(messages) == 0 {
		return "no peer findings available"
	}

	var builder strings.Builder

	for _, message := range messages {
		fmt.Fprintf(&builder, "[%s] %s\n", message.From, message.Content)
	}

	return strings.TrimSpace(builder.String())
}

func stringField(input map[string]any, key string) string {
	value, _ := input[key].(string)

	return value
}
