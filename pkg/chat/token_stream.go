package chat

import (
	"errors"
	"fmt"

	"github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
tokenStream decodes generated token IDs without splitting UTF-8 byte fragments.
*/
type tokenStream struct {
	tokenizer tokenizer.Tokenizer
	tokenIDs  []int
}

/*
newTokenStream creates a tokenizer-backed streaming decoder.
*/
func newTokenStream(modelTokenizer tokenizer.Tokenizer) *tokenStream {
	return &tokenStream{tokenizer: modelTokenizer}
}

/*
Append buffers token IDs and emits text once the buffered byte sequence is valid.
*/
func (stream *tokenStream) Append(tokenIDs []int, emit func(string) error) error {
	if len(tokenIDs) == 0 {
		return nil
	}

	stream.tokenIDs = append(stream.tokenIDs, tokenIDs...)

	return stream.emit(false, emit)
}

/*
Flush emits remaining valid text and reports incomplete byte sequences.
*/
func (stream *tokenStream) Flush(emit func(string) error) error {
	return stream.emit(true, emit)
}

func (stream *tokenStream) emit(final bool, emit func(string) error) error {
	if stream == nil || stream.tokenizer == nil {
		return fmt.Errorf("chat.model: tokenizer stream is not initialized")
	}

	if len(stream.tokenIDs) == 0 {
		return nil
	}

	text, err := stream.tokenizer.Decode(stream.tokenIDs, true)

	if errors.Is(err, tokenizer.ErrInvalidUTF8) && !final {
		return nil
	}

	if errors.Is(err, tokenizer.ErrInvalidUTF8) {
		return fmt.Errorf(
			"chat.model: incomplete tokenizer byte sequence after %d generated tokens: %w",
			len(stream.tokenIDs),
			err,
		)
	}

	if err != nil {
		return err
	}

	stream.tokenIDs = stream.tokenIDs[:0]

	if text == "" {
		return nil
	}

	return emit(text)
}
