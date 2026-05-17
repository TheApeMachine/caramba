package devteam

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	devcfg "github.com/theapemachine/caramba/pkg/config"
)

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
		`git -C /workspace diff -- . && git -C /workspace diff --cached -- .`,
	)

	if err != nil {
		diff = "(could not read diff: " + err.Error() + ")"
	}

	testOutput, _ := sandbox.Exec(`cd /workspace && go test ./... 2>&1 | tail -60`)

	system := reviewer.SystemPrompt()

	userContent := fmt.Sprintf(
		"Feature: %s\n\n%s\n\n--- git diff ---\n%s\n\n--- test output ---\n%s",
		title, description, truncate(diff, 6000), truncate(testOutput, 2000),
	)

	started := time.Now()
	resp, err := reviewer.llm.Chat(reviewer.ctx, ChatRequest{
		System:    system,
		Messages:  []ChatMessage{{Role: "user", Content: userContent}},
		MaxTokens: 1024,
	})

	if err != nil {
		return ReviewVerdict{}, fmt.Errorf("reviewer: %w", err)
	}

	publishChatUsage("reviewer", started, resp)

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

/*
SystemPrompt returns the embedded reviewer prompt.
*/
func (reviewer *Reviewer) SystemPrompt() string {
	return strings.TrimSpace(agentAssets.Reviewer.SystemPrompt)
}
