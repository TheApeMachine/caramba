package devteam

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
	devcfg "github.com/theapemachine/caramba/pkg/config"
)

type staticProvider struct {
	delay time.Duration
}

func (provider *staticProvider) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	if provider.delay > 0 {
		select {
		case <-time.After(provider.delay):
		case <-ctx.Done():
			return ChatResponse{}, ctx.Err()
		}
	}

	return ChatResponse{Content: req.Messages[0].Content}, nil
}

type toolProvider struct {
	mu    sync.Mutex
	calls int
}

type memoryEditor struct {
	hits []SearchResult
}

func (provider *toolProvider) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	provider.mu.Lock()
	defer provider.mu.Unlock()

	provider.calls++

	if provider.calls == 1 {
		return ChatResponse{
			ToolCalls: []ToolCall{{
				ID:   "call-1",
				Name: "search_code",
				Input: map[string]any{
					"pattern":     "Developer",
					"max_results": float64(1),
				},
			}},
		}, nil
	}

	return ChatResponse{Content: req.Messages[len(req.Messages)-1].Content}, nil
}

func (editor *memoryEditor) Search(pattern string, maxResults int) ([]SearchResult, error) {
	return editor.hits, nil
}

func (editor *memoryEditor) View(path string, fromLine, toLine uint32) (string, error) {
	return "viewed " + path, nil
}

func TestSubAgentPoolParse(t *testing.T) {
	Convey("Given a sub-agent tool input", t, func() {
		pool := NewSubAgentPool(context.Background(), devcfg.ProviderConfig{}, nil)

		Convey("It should parse prompt-controlled tasks", func() {
			tasks, err := pool.Parse(map[string]any{
				"tasks": []any{
					map[string]any{
						"name":          "reader",
						"system_prompt": "Read only.",
						"user_prompt":   "Find the agent package.",
					},
				},
			})

			So(err, ShouldBeNil)
			So(len(tasks), ShouldEqual, 1)
			So(tasks[0].Name, ShouldEqual, "reader")
			So(tasks[0].SystemPrompt, ShouldEqual, "Read only.")
			So(tasks[0].UserPrompt, ShouldEqual, "Find the agent package.")
		})

		Convey("It should reject missing prompts", func() {
			_, err := pool.Parse(map[string]any{
				"tasks": []any{map[string]any{"name": "broken"}},
			})

			So(err, ShouldNotBeNil)
		})
	})
}

func TestSubAgentPoolRun(t *testing.T) {
	Convey("Given a sub-agent pool with multiple tasks", t, func() {
		pool := NewSubAgentPool(context.Background(), devcfg.ProviderConfig{}, nil)
		pool.newProvider = func(devcfg.ProviderConfig) Provider {
			return &staticProvider{delay: 25 * time.Millisecond}
		}

		Convey("It should execute tasks in parallel and preserve result order", func() {
			tasks := []SubAgentTask{
				{Name: "first", SystemPrompt: "sys", UserPrompt: "one"},
				{Name: "second", SystemPrompt: "sys", UserPrompt: "two"},
				{Name: "third", SystemPrompt: "sys", UserPrompt: "three"},
			}

			started := time.Now()
			results := pool.Run(tasks)

			So(time.Since(started), ShouldBeLessThan, 75*time.Millisecond)
			So(results[0].Name, ShouldEqual, "first")
			So(results[0].Output, ShouldEqual, "one")
			So(results[1].Name, ShouldEqual, "second")
			So(results[1].Output, ShouldEqual, "two")
			So(results[2].Name, ShouldEqual, "third")
			So(results[2].Output, ShouldEqual, "three")
		})
	})
}

func TestSubAgentRun(t *testing.T) {
	Convey("Given a sub-agent with read-only tools", t, func() {
		pool := NewSubAgentPool(context.Background(), devcfg.ProviderConfig{}, nil)
		coordination := pool.NewCoordination([]SubAgentTask{{Name: "reader"}})
		defer coordination.Close()

		agent := &subAgent{
			ctx:          context.Background(),
			name:         "reader",
			provider:     &toolProvider{},
			coordination: coordination,
			editor: &memoryEditor{hits: []SearchResult{{
				Path: "pkg/devteam/agent.go",
				Line: 115,
				Text: "Developer is an AI agent",
			}}},
		}

		Convey("It should execute search_code tool calls and return the final answer", func() {
			output, err := agent.Run("system", "find developer")

			So(err, ShouldBeNil)
			So(output, ShouldContainSubstring, "pkg/devteam/agent.go")
		})
	})
}

func TestSubAgentCoordination(t *testing.T) {
	Convey("Given sibling sub-agents in one coordination room", t, func() {
		pool := NewSubAgentPool(context.Background(), devcfg.ProviderConfig{}, nil)
		coordination := pool.NewCoordination([]SubAgentTask{
			{Name: "reader"},
			{Name: "reviewer"},
		})

		defer coordination.Close()

		Convey("It should list subscribed peers", func() {
			peers := coordination.Peers()

			So(peers, ShouldContainSubstring, "reader")
			So(peers, ShouldContainSubstring, "reviewer")
		})

		Convey("It should broadcast findings between peers", func() {
			So(coordination.Publish("reader", "pkg/devteam/subagent.go owns coordination"), ShouldEqual, "published")

			messages := coordination.Read("reviewer", 10)

			So(messages, ShouldContainSubstring, "reader")
			So(messages, ShouldContainSubstring, "pkg/devteam/subagent.go")
		})

		Convey("It should not return a peer's own finding", func() {
			So(coordination.Publish("reader", "self note"), ShouldEqual, "published")

			messages := coordination.Read("reader", 10)

			So(messages, ShouldNotContainSubstring, "self note")
		})
	})
}

func BenchmarkSubAgentPoolParse(b *testing.B) {
	pool := NewSubAgentPool(context.Background(), devcfg.ProviderConfig{}, nil)
	input := map[string]any{
		"tasks": []any{map[string]any{
			"name":          "reader",
			"system_prompt": "Read only.",
			"user_prompt":   "Find agent code.",
		}},
	}

	for b.Loop() {
		_, _ = pool.Parse(input)
	}
}

func BenchmarkSubAgentPoolFormat(b *testing.B) {
	pool := NewSubAgentPool(context.Background(), devcfg.ProviderConfig{}, nil)
	results := make([]SubAgentResult, 16)

	for index := range results {
		results[index] = SubAgentResult{
			Name:   fmt.Sprintf("agent-%d", index),
			Output: strings.Repeat("result ", 8),
		}
	}

	for b.Loop() {
		_ = pool.Format(results)
	}
}

func BenchmarkSubAgentCoordinationPublishRead(b *testing.B) {
	pool := NewSubAgentPool(context.Background(), devcfg.ProviderConfig{}, nil)
	coordination := pool.NewCoordination([]SubAgentTask{
		{Name: "reader"},
		{Name: "reviewer"},
	})

	defer coordination.Close()

	for b.Loop() {
		coordination.Publish("reader", "finding")
		_ = coordination.Read("reviewer", 1)
	}
}
