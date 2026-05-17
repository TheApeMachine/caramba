package tokenize

import (
	"errors"
	"fmt"

	"github.com/theapemachine/caramba/pkg/runtime/op"
	"github.com/theapemachine/caramba/pkg/runtime/program"
	"github.com/theapemachine/caramba/pkg/runtime/state"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
Encode tokenizes Inputs["text"] using the tokenizer asset referenced
by Inputs["tokenizer"] and writes the resulting []int to
Outputs["tokens"].
*/
type Encode struct{}

func (Encode) Execute(execContext op.Context) error {
	step := execContext.Step()
	tokenizerRef, ok := step.Inputs["tokenizer"]

	if !ok || tokenizerRef.Namespace != program.NamespaceTokenizer {
		return fmt.Errorf("tokenizer.encode: inputs.tokenizer must reference the tokenizer namespace")
	}

	textRef, ok := step.Inputs["text"]

	if !ok {
		return fmt.Errorf("tokenizer.encode: missing input 'text'")
	}

	outputRef, ok := step.Outputs["tokens"]

	if !ok {
		return fmt.Errorf("tokenizer.encode: missing output 'tokens'")
	}

	tokenizerInstance, err := execContext.Tokenizer(tokenizerRef.Name)

	if err != nil {
		return err
	}

	value, err := execContext.Resolve(textRef)

	if err != nil {
		return err
	}

	text, ok := value.(string)

	if !ok {
		return fmt.Errorf("tokenizer.encode: text must be string, got %T", value)
	}

	tokens, err := tokenizerInstance.Encode(text)

	if err != nil {
		return fmt.Errorf("tokenizer.encode: %w", err)
	}

	return execContext.Bind(outputRef, tokens)
}

/*
StreamDecode consumes one token, appends it to a TokenStream state
object, and emits decoded text whenever the buffered bytes form a
valid UTF-8 sequence. When the bytes are incomplete the op binds an
empty string to the output so the caller's emit step is a no-op.
*/
type StreamDecode struct{}

func (StreamDecode) Execute(execContext op.Context) error {
	step := execContext.Step()
	tokenizerRef, ok := step.Inputs["tokenizer"]

	if !ok || tokenizerRef.Namespace != program.NamespaceTokenizer {
		return fmt.Errorf("tokenizer.stream_decode: inputs.tokenizer must reference the tokenizer namespace")
	}

	streamRef, ok := step.Inputs["stream"]

	if !ok || streamRef.Namespace != program.NamespaceState {
		return fmt.Errorf("tokenizer.stream_decode: inputs.stream must reference a state object")
	}

	tokenRef, ok := step.Inputs["token"]

	if !ok {
		return fmt.Errorf("tokenizer.stream_decode: missing input 'token'")
	}

	textRef, ok := step.Outputs["text"]

	if !ok {
		return fmt.Errorf("tokenizer.stream_decode: missing output 'text'")
	}

	tokenizerInstance, err := execContext.Tokenizer(tokenizerRef.Name)

	if err != nil {
		return err
	}

	streamInstance, err := streamState(execContext, streamRef.Name)

	if err != nil {
		return err
	}

	tokenValue, err := execContext.Resolve(tokenRef)

	if err != nil {
		return err
	}

	tokenID, err := asInt(tokenValue)

	if err != nil {
		return fmt.Errorf("tokenizer.stream_decode: %w", err)
	}

	streamInstance.Append(tokenID)
	skipSpecial, _ := step.Config["skip_special_tokens"].(bool)

	text, err := tokenizerInstance.Decode(streamInstance.Pending(), skipSpecial)

	if errors.Is(err, tokenizer.ErrInvalidUTF8) {
		return execContext.Bind(textRef, "")
	}

	if err != nil {
		return fmt.Errorf("tokenizer.stream_decode: %w", err)
	}

	streamInstance.Clear()

	return execContext.Bind(textRef, text)
}

/*
StreamFlush decodes whatever bytes remain in the TokenStream buffer
and writes them to Outputs["text"]. An invalid-UTF-8 error at flush
time is fatal — incomplete sequences at the end of a generation are
a real bug, not a transient buffer state.
*/
type StreamFlush struct{}

func (StreamFlush) Execute(execContext op.Context) error {
	step := execContext.Step()
	tokenizerRef, ok := step.Inputs["tokenizer"]

	if !ok || tokenizerRef.Namespace != program.NamespaceTokenizer {
		return fmt.Errorf("tokenizer.stream_flush: inputs.tokenizer must reference the tokenizer namespace")
	}

	streamRef, ok := step.Inputs["stream"]

	if !ok || streamRef.Namespace != program.NamespaceState {
		return fmt.Errorf("tokenizer.stream_flush: inputs.stream must reference a state object")
	}

	textRef, ok := step.Outputs["text"]

	if !ok {
		return fmt.Errorf("tokenizer.stream_flush: missing output 'text'")
	}

	tokenizerInstance, err := execContext.Tokenizer(tokenizerRef.Name)

	if err != nil {
		return err
	}

	streamInstance, err := streamState(execContext, streamRef.Name)

	if err != nil {
		return err
	}

	pending := streamInstance.Pending()

	if len(pending) == 0 {
		return execContext.Bind(textRef, "")
	}

	skipSpecial, _ := step.Config["skip_special_tokens"].(bool)
	text, err := tokenizerInstance.Decode(pending, skipSpecial)

	if err != nil {
		return fmt.Errorf("tokenizer.stream_flush: %w", err)
	}

	streamInstance.Clear()

	return execContext.Bind(textRef, text)
}

func streamState(execContext op.Context, stateID string) (*state.TokenStream, error) {
	instance, err := execContext.State(stateID)

	if err != nil {
		return nil, err
	}

	stream, ok := instance.(*state.TokenStream)

	if !ok {
		return nil, fmt.Errorf("tokenizer/stream: state %q is type %q, want token_stream", stateID, instance.Type())
	}

	return stream, nil
}

/*
Decode reverses Encode. Inputs["tokens"] is a []int; Outputs["text"]
receives the decoded string.
*/
type Decode struct{}

func (Decode) Execute(execContext op.Context) error {
	step := execContext.Step()
	tokenizerRef, ok := step.Inputs["tokenizer"]

	if !ok || tokenizerRef.Namespace != program.NamespaceTokenizer {
		return fmt.Errorf("tokenizer.decode: inputs.tokenizer must reference the tokenizer namespace")
	}

	tokensRef, ok := step.Inputs["tokens"]

	if !ok {
		return fmt.Errorf("tokenizer.decode: missing input 'tokens'")
	}

	outputRef, ok := step.Outputs["text"]

	if !ok {
		return fmt.Errorf("tokenizer.decode: missing output 'text'")
	}

	tokenizerInstance, err := execContext.Tokenizer(tokenizerRef.Name)

	if err != nil {
		return err
	}

	value, err := execContext.Resolve(tokensRef)

	if err != nil {
		return err
	}

	tokens, err := asIntSlice(value)

	if err != nil {
		return fmt.Errorf("tokenizer.decode: %w", err)
	}

	skipSpecial, _ := step.Config["skip_special_tokens"].(bool)

	text, err := tokenizerInstance.Decode(tokens, skipSpecial)

	if err != nil {
		return fmt.Errorf("tokenizer.decode: %w", err)
	}

	return execContext.Bind(outputRef, text)
}

func asIntSlice(value any) ([]int, error) {
	switch typed := value.(type) {
	case []int:
		return append([]int(nil), typed...), nil
	case []any:
		out := make([]int, len(typed))

		for index, element := range typed {
			cast, err := asInt(element)

			if err != nil {
				return nil, fmt.Errorf("element %d: %w", index, err)
			}

			out[index] = cast
		}

		return out, nil
	}

	return nil, fmt.Errorf("expected []int, got %T", value)
}

func asInt(value any) (int, error) {
	switch typed := value.(type) {
	case int:
		return typed, nil
	case int32:
		return int(typed), nil
	case int64:
		if typed > int64(maxPlatformInt) || typed < int64(minPlatformInt) {
			return 0, fmt.Errorf("expected integer, got int64 %d out of int range", typed)
		}

		return int(typed), nil
	case float64:
		if typed > float64(maxPlatformInt) || typed < float64(minPlatformInt) {
			return 0, fmt.Errorf("expected integer, got float64 %g out of int range", typed)
		}

		return int(typed), nil
	}

	return 0, fmt.Errorf("expected integer, got %T", value)
}

const (
	maxPlatformInt = int(^uint(0) >> 1)
	minPlatformInt = -maxPlatformInt - 1
)

func init() {
	op.Default.MustRegister("tokenizer.encode", Encode{})
	op.Default.MustRegister("tokenizer.decode", Decode{})
	op.Default.MustRegister("tokenizer.stream_decode", StreamDecode{})
	op.Default.MustRegister("tokenizer.stream_flush", StreamFlush{})
}
