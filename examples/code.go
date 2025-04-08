package examples

import (
	"bytes"
	"context"
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
	"github.com/theapemachine/caramba/pkg/datura"
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
	planner agent.Agent
	dev     agent.Agent
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
	err = errnie.RunSafely(func() {
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

		client := agent.AgentToClient(code.planner)

		future, release := client.Send(context.Background(), func(params agent.RPC_send_Params) error {
			return params.SetArtifact(datura.New(
				datura.WithBytes(
					message.New(
						message.WithRole("user"),
						message.WithContent(strings.Join([]string{
							"Write a plan for an innovative research and development project",
							"focused on the development of new AI architectures.",
							"The goal is to increase intelligence, while reducing cost and energy consumption.",
							"The target is to create a new AI architecture that is more intelligent than any",
							"existing architecture, while able to run on consumer hardware.",
						}, " ")),
					).Bytes(),
				),
			))
		})

		defer release()

		out, err := future.Struct()
		if errnie.Error(err) != nil {
			errnie.Fatal(err)
		}

		payload := errnie.Try(out.Out())
		msg := message.New(
			message.WithBytes(payload.Bytes()),
		)

		errnie.Try(io.Copy(
			os.Stdout,
			bytes.NewBufferString(errnie.Try(msg.Content())),
		))

		code.wait()
	})

	return errnie.New(
		errnie.WithError(err),
		errnie.WithMessage("failed to run agent framework test"),
	)
}

func (code *Code) wait() {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Wait for signal or timeout
	select {
	case sig := <-sigChan:
		errnie.Info("\nReceived signal %v, shutting down...\n", sig)
		code.cancel()
	case <-time.After(60 * time.Second):
		errnie.Warn("Timeout reached, shutting down...")
		code.cancel()
	}

	errnie.Success("Shutdown complete")
}
