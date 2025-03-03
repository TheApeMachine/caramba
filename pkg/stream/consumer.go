package stream

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"
)

// PartialJSONParser accumulates incoming chunks in a buffer.
// Then we use a json.Decoder to parse tokens out of that buffer.
type Consumer struct {
	buf *bytes.Buffer
	dec *json.Decoder
}

// NewConsumer constructs a parser with an empty buffer.
func NewConsumer() *Consumer {
	buf := new(bytes.Buffer)
	dec := json.NewDecoder(buf)
	return &Consumer{buf, dec}
}

// Feed appends the chunk to our buffer and tries to parse tokens out of it.
// We print each token as we find it. If we run out of data mid‐token, we keep
// the leftover in the buffer. On the next Feed, we'll resume from there.
func (c *Consumer) Feed(chunk string) string {
	// Write new chunk data
	_, _ = c.buf.WriteString(chunk)

	var result strings.Builder

	// Repeatedly try to parse the next token
	for {
		token, err := c.dec.Token()
		if err != nil {
			// If it's EOF or an unexpected EOF, that's likely incomplete JSON
			if errors.Is(err, io.EOF) {
				// We'll wait for more data
				break
			}
			// The json.Decoder does not have a "safe" partial parse mode,
			// so if there's a genuine syntax error, we might be stuck.
			// In a real app, you might handle or return it differently.
			break
		}

		// Add token to result
		if str, ok := token.(string); ok {
			result.WriteString(str)
		}
	}

	return result.String()
}
