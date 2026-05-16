package chat

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"strings"
)

const scannerCapacity = 1024 * 1024

/*
Session owns the terminal chat loop.
It is intentionally independent from model execution so the same prompt
experience can stream from preview, local, or distributed generators.
*/
type Session struct {
	ctx       context.Context
	input     io.Reader
	output    io.Writer
	generator Generator
	config    SessionConfig
}

/*
SessionConfig contains terminal-facing chat settings.
*/
type SessionConfig struct {
	Runtime         string
	Backend         string
	Model           string
	UserPrompt      string
	AssistantPrompt string
	ShowBanner      bool
}

/*
NewSession instantiates a terminal chat session.
*/
func NewSession(
	ctx context.Context,
	input io.Reader,
	output io.Writer,
	generator Generator,
	config SessionConfig,
) *Session {
	return &Session{
		ctx:       ctx,
		input:     input,
		output:    output,
		generator: generator,
		config:    config.WithDefaults(),
	}
}

/*
WithDefaults fills in the stable terminal defaults.
*/
func (config SessionConfig) WithDefaults() SessionConfig {
	if config.Runtime == "" {
		config.Runtime = "preview"
	}

	if config.UserPrompt == "" {
		config.UserPrompt = "you> "
	}

	if config.AssistantPrompt == "" {
		config.AssistantPrompt = "caramba> "
	}

	return config
}

/*
Run starts the interactive prompt loop.
*/
func (session *Session) Run() error {
	if session.config.ShowBanner {
		if err := session.writeBanner(); err != nil {
			return err
		}
	}

	scanner := bufio.NewScanner(session.input)
	scanner.Buffer(make([]byte, 0, 4096), scannerCapacity)

	for {
		if _, err := fmt.Fprint(session.output, session.config.UserPrompt); err != nil {
			return err
		}

		if !scanner.Scan() {
			break
		}

		prompt := strings.TrimSpace(scanner.Text())

		if prompt == "" {
			continue
		}

		handled, err := session.command(prompt)

		if err != nil {
			return err
		}

		if handled {
			continue
		}

		if session.exit(prompt) {
			return nil
		}

		if err := session.respond(prompt); err != nil {
			return err
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("chat: read prompt: %w", err)
	}

	return nil
}

/*
RunPrompt streams one response and exits.
*/
func (session *Session) RunPrompt(prompt string) error {
	prompt = strings.TrimSpace(prompt)

	if prompt == "" {
		return nil
	}

	if session.config.ShowBanner {
		if err := session.writeBanner(); err != nil {
			return err
		}
	}

	return session.respond(prompt)
}

func (session *Session) respond(prompt string) error {
	if session.generator == nil {
		return fmt.Errorf("chat: generator is required")
	}

	if _, err := fmt.Fprint(session.output, session.config.AssistantPrompt); err != nil {
		return err
	}

	err := session.generator.Generate(session.ctx, prompt, func(chunk string) error {
		_, writeErr := io.WriteString(session.output, chunk)

		return writeErr
	})

	if err != nil {
		return err
	}

	_, err = fmt.Fprintln(session.output)

	return err
}

func (session *Session) command(prompt string) (bool, error) {
	if prompt != "/help" {
		return false, nil
	}

	_, err := fmt.Fprintln(session.output, "commands: /help /exit /quit")

	return true, err
}

func (session *Session) exit(prompt string) bool {
	return prompt == "/exit" || prompt == "/quit"
}

func (session *Session) writeBanner() error {
	model := strings.TrimSpace(session.config.Model)
	backend := strings.TrimSpace(session.config.Backend)

	if backend != "" && model == "" {
		_, err := fmt.Fprintf(
			session.output,
			"caramba chat runtime=%s backend=%s\n",
			session.config.Runtime,
			backend,
		)

		return err
	}

	if model == "" {
		_, err := fmt.Fprintf(session.output, "caramba chat runtime=%s\n", session.config.Runtime)

		return err
	}

	if backend != "" {
		_, err := fmt.Fprintf(
			session.output,
			"caramba chat runtime=%s backend=%s model=%s\n",
			session.config.Runtime,
			backend,
			model,
		)

		return err
	}

	_, err := fmt.Fprintf(session.output, "caramba chat runtime=%s model=%s\n", session.config.Runtime, model)

	return err
}
