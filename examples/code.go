package examples

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/theapemachine/caramba/pkg/ai/agent"
	"github.com/theapemachine/caramba/pkg/ai/message"
	"github.com/theapemachine/caramba/pkg/ai/prompt"
	prvdr "github.com/theapemachine/caramba/pkg/ai/provider"
	"github.com/theapemachine/caramba/pkg/ai/tool"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tools"
	"github.com/theapemachine/caramba/pkg/tweaker"
	"github.com/theapemachine/caramba/pkg/twoface"
	"github.com/theapemachine/caramba/pkg/utils"
)

// Code represents a test setup for the agent framework
type Code struct {
	ctx     context.Context
	cancel  context.CancelFunc
	hub     *twoface.Hub
	planner *agent.AgentBuilder
	dev     *agent.AgentBuilder
}

// NewCode creates a new test setup for the agent framework
func NewCode() *Code {
	ctx, cancel := context.WithCancel(context.Background())

	return &Code{
		ctx:    ctx,
		cancel: cancel,
		hub:    twoface.NewHub(),
	}
}

// Run executes the test setup
func (code *Code) Run() (err error) {
	errnie.Info("Agent framework test")

	systemTool := tool.New(
		tool.WithMCPTool(tools.NewSystemTool().ToMCP()...),
	)

	code.planner = agent.New(
		agent.WithName(utils.GenerateName()),
		agent.WithRole("planner"),
		agent.WithModel(tweaker.GetModel("openai")),
		agent.WithProvider(
			prvdr.New(
				prvdr.WithAIProvider("openai", provider.NewOpenAIProvider()),
			),
		),
		agent.WithTransport(code.hub.NewTransport()),
		agent.WithTools(systemTool),
		agent.WithPrompt("system", prompt.New(
			prompt.WithFragments(
				prompt.NewFragmentBuilder(
					prompt.WithBuiltin("planner"),
				),
			),
		)),
	)

	code.dev = agent.New(
		agent.WithName(utils.GenerateName()),
		agent.WithRole("developer"),
		agent.WithModel(tweaker.GetModel("openai")),
		agent.WithProvider(
			prvdr.New(
				prvdr.WithAIProvider("openai", provider.NewOpenAIProvider()),
			),
		),
		agent.WithTransport(code.hub.NewTransport()),
		agent.WithTools(systemTool),
		agent.WithPrompt("system", prompt.New(
			prompt.WithFragments(
				prompt.NewFragmentBuilder(
					prompt.WithBuiltin("developer"),
				),
			),
		)),
	)

	out := code.planner.Send(message.New(
		message.WithRole("user"),
		message.WithContent(strings.Join([]string{
			"Write a plan for an innovative research and development project",
			"focused on the development of new AI architectures.",
			"The goal is to increase intelligence, while reducing cost and energy consumption.",
			"The target is to create a new AI architecture that is more intelligent than any",
			"existing architecture, while able to run on consumer hardware.",
		}, " ")),
	))

	payload := errnie.Try(out.Payload())
	errnie.Info("payload", "payload", string(payload))
	msg := message.New()

	if _, err = io.Copy(msg, bytes.NewBuffer(payload)); errnie.Error(err) != nil {
		return errnie.Error(err)
	}

	if _, err = io.Copy(
		os.Stdout,
		bytes.NewBufferString(errnie.Try(msg.Message.Content())),
	); errnie.Error(err) != nil {
		return errnie.Error(err)
	}

	code.wait()
	return nil
}

func (code *Code) wait() {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Wait for signal or timeout
	select {
	case sig := <-sigChan:
		fmt.Printf("\nReceived signal %v, shutting down...\n", sig)
		code.cancel()
	case <-time.After(60 * time.Second):
		fmt.Println("Timeout reached, shutting down...")
		code.cancel()
	}

	fmt.Println("Shutdown complete")
}
