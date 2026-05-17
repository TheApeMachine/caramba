package io

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"

	"github.com/theapemachine/caramba/pkg/runtime/op"
	"github.com/theapemachine/caramba/pkg/runtime/program"
)

/*
ReadLine reads one line from the executor's stdin and writes it to
Outputs["text"]. A trailing newline is stripped.
*/
type ReadLine struct {
	mu      sync.Mutex
	readers map[io.Reader]*bufio.Reader
}

func newReadLine() *ReadLine {
	return &ReadLine{readers: map[io.Reader]*bufio.Reader{}}
}

func (readLine *ReadLine) Execute(execContext op.Context) error {
	target, ok := execContext.Step().Outputs["text"]

	if !ok {
		return fmt.Errorf("io.read_line: missing output 'text'")
	}

	stdin := execContext.Stdin()
	readLine.mu.Lock()
	reader, ok := readLine.readers[stdin]

	if !ok {
		reader = bufio.NewReader(stdin)
		readLine.readers[stdin] = reader
	}

	// Hold readLine.mu through the blocking read so two concurrent
	// callers sharing the same stdin cannot read interleaved bytes
	// from the same bufio.Reader. The runtime is single-threaded
	// today; this is defence for future concurrent executors.
	line, err := reader.ReadString('\n')
	readLine.mu.Unlock()

	// EOF on an empty line is the documented "session is over"
	// signal: raising op.ErrBreak lets the enclosing loop
	// (typically the chat manifest's outer control.loop_until)
	// exit cleanly without surfacing the EOF as a program error.
	if errors.Is(err, io.EOF) && line == "" {
		return op.ErrBreak
	}

	if err != nil && line == "" {
		return fmt.Errorf("io.read_line: %w", err)
	}

	trimmed := strings.TrimRight(line, "\r\n")

	return execContext.Bind(target, trimmed)
}

/*
EmitText writes Inputs["text"] to stdout. The text is written as-is;
callers that want a newline append it in the program.
*/
type EmitText struct{}

func (EmitText) Execute(execContext op.Context) error {
	step := execContext.Step()

	// Manifests that emit a fixed prompt or banner can put the
	// literal in config.text so they do not have to thread a local
	// ValueRef through value.assign first. Inputs.text still works
	// for dynamically resolved strings (decoded tokens, etc.).
	if literal, ok := step.Config["text"].(string); ok {
		if _, err := execContext.Stdout().Write([]byte(literal)); err != nil {
			return fmt.Errorf("io.emit_text: %w", err)
		}

		return nil
	}

	source, ok := step.Inputs["text"]

	if !ok {
		return fmt.Errorf("io.emit_text: missing input 'text'")
	}

	value, err := execContext.Resolve(source)

	if err != nil {
		return err
	}

	text, ok := value.(string)

	if !ok {
		return fmt.Errorf("io.emit_text: expected string, got %T", value)
	}

	if _, err := execContext.Stdout().Write([]byte(text)); err != nil {
		return fmt.Errorf("io.emit_text: %w", err)
	}

	return nil
}

/*
EmitToken decodes Inputs["token"] using the tokenizer named by
Inputs["tokenizer"] (an asset reference) and writes the resulting
text to stdout. This is the streaming-decode primitive the chat
runtime uses inside its generation loop.
*/
type EmitToken struct{}

func (EmitToken) Execute(execContext op.Context) error {
	step := execContext.Step()
	tokenRef, ok := step.Inputs["token"]

	if !ok {
		return fmt.Errorf("io.emit_token: missing input 'token'")
	}

	tokenizerRef, ok := step.Inputs["tokenizer"]

	if !ok || tokenizerRef.Namespace != program.NamespaceTokenizer {
		return fmt.Errorf("io.emit_token: tokenizer must be a tokenizer asset reference")
	}

	rawToken, err := execContext.Resolve(tokenRef)

	if err != nil {
		return err
	}

	tokenID, err := asInt(rawToken)

	if err != nil {
		return fmt.Errorf("io.emit_token: %w", err)
	}

	tokenizerInstance, err := execContext.Tokenizer(tokenizerRef.Name)

	if err != nil {
		return err
	}

	text, err := tokenizerInstance.Decode([]int{tokenID}, false)

	if err != nil {
		return fmt.Errorf("io.emit_token: %w", err)
	}

	if _, err := execContext.Stdout().Write([]byte(text)); err != nil {
		return fmt.Errorf("io.emit_token: %w", err)
	}

	return nil
}

func asInt(value any) (int, error) {
	switch typed := value.(type) {
	case int:
		return typed, nil
	case int32:
		return int(typed), nil
	case int64:
		return int(typed), nil
	case float64:
		return int(typed), nil
	}

	return 0, fmt.Errorf("expected integer, got %T", value)
}

func init() {
	op.Default.MustRegister("io.read_line", newReadLine())
	op.Default.MustRegister("io.emit_text", EmitText{})
	op.Default.MustRegister("io.emit_token", EmitToken{})
}
