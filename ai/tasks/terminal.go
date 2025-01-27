package tasks

import (
	"io"
	"strings"

	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/tools"
)

// Global singleton instance of Terminal task to maintain state
var globalTerminal *Terminal

type Terminal struct {
	container *tools.Container
	conn      io.ReadWriteCloser
	isActive  bool
}

func NewTerminal() *Terminal {
	if globalTerminal == nil {
		globalTerminal = &Terminal{
			container: tools.NewContainer(),
		}
	}
	return globalTerminal
}

func (task *Terminal) Execute(ctx *drknow.Context, accumulator *stream.Accumulator, args map[string]any) {
	// If terminal is not active, initialize it
	if !task.isActive {
		if err := task.container.Initialize(); err != nil {
			ctx.AddMessage(
				provider.NewMessage(
					provider.RoleAssistant,
					err.Error(),
				),
			)
			return
		}

		bridge := &containerBridge{
			context: ctx,
		}
		if err := task.container.Connect(ctx.Identity.Ctx, bridge); err != nil {
			ctx.AddMessage(
				provider.NewMessage(
					provider.RoleAssistant,
					err.Error(),
				),
			)
			return
		}
		task.conn = bridge
		task.isActive = true

		// Send initial message
		ctx.AddMessage(
			provider.NewMessage(
				provider.RoleAssistant,
				"Terminal session started. Ready for commands.",
			),
		)
		return
	}

	// Get the last message from the context
	messages := ctx.Identity.Params.Thread.Messages
	if len(messages) == 0 {
		return
	}
	lastMsg := messages[len(messages)-1]

	// Skip if this is a system message or iteration marker
	if lastMsg.Role != provider.RoleAssistant ||
		strings.Contains(lastMsg.Content, "Terminal session started") ||
		strings.Contains(lastMsg.Content, "iteration") {
		return
	}

	// Execute command
	if _, err := task.conn.Write([]byte(lastMsg.Content + "\n")); err != nil {
		ctx.AddMessage(
			provider.NewMessage(
				provider.RoleAssistant,
				err.Error(),
			),
		)
		return
	}

	// Read response with timeout
	buf := make([]byte, 4096)
	n, err := task.conn.Read(buf)
	if err != nil && err != io.EOF {
		ctx.AddMessage(
			provider.NewMessage(
				provider.RoleAssistant,
				err.Error(),
			),
		)
		return
	}

	if n > 0 {
		ctx.AddMessage(
			provider.NewMessage(
				provider.RoleAssistant,
				string(buf[:n]),
			),
		)
	}
}

type containerBridge struct {
	context *drknow.Context
}

func (b *containerBridge) Read(p []byte) (n int, err error) {
	messages := b.context.Identity.Params.Thread.Messages
	if len(messages) == 0 {
		return 0, io.EOF
	}

	lastMsg := messages[len(messages)-1]
	if lastMsg.Role != provider.RoleAssistant {
		return 0, io.EOF
	}

	cmd := []byte(lastMsg.Content + "\n")
	n = copy(p, cmd)
	return n, nil
}

func (b *containerBridge) Write(p []byte) (n int, err error) {
	b.context.AddMessage(
		provider.NewMessage(
			provider.RoleAssistant,
			string(p),
		),
	)
	return len(p), nil
}

func (b *containerBridge) Close() error {
	return nil
}
