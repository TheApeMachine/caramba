package state

import (
	"context"
	"encoding/binary"
	"fmt"
	"sync"
)

/*
TokenStream is a streaming-decode buffer for byte-level tokenizers
where a single token can be a partial UTF-8 sequence. The runtime's
tokenizer.stream_decode op appends tokens here and asks the
tokenizer to decode the buffered window; only when the result is
valid UTF-8 does the op flush the bytes and clear the buffer.
*/
type TokenStream struct {
	mu      sync.Mutex
	id      string
	pending []int
}

func newTokenStream(id string) *TokenStream {
	return &TokenStream{id: id}
}

func newTokenStreamFromConfig(id string, config map[string]any) (State, error) {
	return newTokenStream(id), nil
}

func (tokenStream *TokenStream) ID() string {
	return tokenStream.id
}

func (tokenStream *TokenStream) Type() string {
	return "token_stream"
}

func (tokenStream *TokenStream) Reset(ctx context.Context) error {
	tokenStream.mu.Lock()
	defer tokenStream.mu.Unlock()

	tokenStream.pending = tokenStream.pending[:0]

	return nil
}

/*
Append adds a token ID to the pending buffer.
*/
func (tokenStream *TokenStream) Append(tokenID int) {
	tokenStream.mu.Lock()
	defer tokenStream.mu.Unlock()

	tokenStream.pending = append(tokenStream.pending, tokenID)
}

/*
Pending returns a copy of the currently buffered token IDs.
*/
func (tokenStream *TokenStream) Pending() []int {
	tokenStream.mu.Lock()
	defer tokenStream.mu.Unlock()

	return append([]int(nil), tokenStream.pending...)
}

/*
Clear empties the pending buffer.
*/
func (tokenStream *TokenStream) Clear() {
	tokenStream.mu.Lock()
	defer tokenStream.mu.Unlock()

	tokenStream.pending = tokenStream.pending[:0]
}

func (tokenStream *TokenStream) Snapshot(ctx context.Context) (Snapshot, error) {
	pending := tokenStream.Pending()
	payload := make([]byte, 8*len(pending))

	for index, token := range pending {
		binary.LittleEndian.PutUint64(payload[index*8:], uint64(int64(token)))
	}

	return Snapshot{
		StateID: tokenStream.id,
		Type:    tokenStream.Type(),
		Schema:  "int64-le",
		Payload: payload,
	}, nil
}

func (tokenStream *TokenStream) Restore(ctx context.Context, snapshot Snapshot) error {
	if snapshot.Schema != "int64-le" {
		return fmt.Errorf("token_stream: unsupported snapshot schema %q", snapshot.Schema)
	}

	if len(snapshot.Payload)%8 != 0 {
		return fmt.Errorf("token_stream: payload not a multiple of 8")
	}

	count := len(snapshot.Payload) / 8
	pending := make([]int, count)

	for index := range pending {
		pending[index] = int(int64(binary.LittleEndian.Uint64(snapshot.Payload[index*8:])))
	}

	tokenStream.mu.Lock()
	tokenStream.pending = pending
	tokenStream.mu.Unlock()

	return nil
}

func (tokenStream *TokenStream) Inspect(ctx context.Context) (Inspection, error) {
	return Inspection{
		StateID: tokenStream.id,
		Type:    tokenStream.Type(),
		Values: map[string]any{
			"pending": len(tokenStream.Pending()),
		},
	}, nil
}

func init() {
	Default.MustRegister("token_stream", newTokenStreamFromConfig)
}
