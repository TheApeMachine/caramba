package state

import (
	"context"
	"encoding/binary"
	"fmt"
	"sync"
)

/*
TokenBuffer is the canonical history state for token-decode loops.
Chat history, prompt prefixes, beam candidates, and reference
sequences all share this type so swapping them is a manifest-level
choice rather than a code change.
*/
type TokenBuffer struct {
	mu     sync.Mutex
	id     string
	tokens []int
}

func newTokenBuffer(id string) *TokenBuffer {
	return &TokenBuffer{id: id}
}

func newTokenBufferFromConfig(id string, config map[string]any) (State, error) {
	buffer := newTokenBuffer(id)
	initial, err := intSliceFromConfig(config, "initial")

	if err != nil {
		return nil, err
	}

	buffer.tokens = initial

	return buffer, nil
}

func (tokenBuffer *TokenBuffer) ID() string {
	return tokenBuffer.id
}

func (tokenBuffer *TokenBuffer) Type() string {
	return "token_buffer"
}

func (tokenBuffer *TokenBuffer) Reset(ctx context.Context) error {
	tokenBuffer.mu.Lock()
	defer tokenBuffer.mu.Unlock()

	tokenBuffer.tokens = tokenBuffer.tokens[:0]

	return nil
}

/*
Append adds a single token to the tail of the buffer.
*/
func (tokenBuffer *TokenBuffer) Append(token int) {
	tokenBuffer.mu.Lock()
	defer tokenBuffer.mu.Unlock()

	tokenBuffer.tokens = append(tokenBuffer.tokens, token)
}

/*
Extend appends a slice of tokens.
*/
func (tokenBuffer *TokenBuffer) Extend(tokens []int) {
	if len(tokens) == 0 {
		return
	}

	tokenBuffer.mu.Lock()
	defer tokenBuffer.mu.Unlock()

	tokenBuffer.tokens = append(tokenBuffer.tokens, tokens...)
}

/*
Tokens returns a copy of the current token sequence.
*/
func (tokenBuffer *TokenBuffer) Tokens() []int {
	tokenBuffer.mu.Lock()
	defer tokenBuffer.mu.Unlock()

	return append([]int(nil), tokenBuffer.tokens...)
}

/*
Length returns the number of tokens currently buffered.
*/
func (tokenBuffer *TokenBuffer) Length() int {
	tokenBuffer.mu.Lock()
	defer tokenBuffer.mu.Unlock()

	return len(tokenBuffer.tokens)
}

func (tokenBuffer *TokenBuffer) Snapshot(ctx context.Context) (Snapshot, error) {
	tokens := tokenBuffer.Tokens()
	payload := make([]byte, 8*len(tokens))

	for index, token := range tokens {
		binary.LittleEndian.PutUint64(payload[index*8:], uint64(int64(token)))
	}

	return Snapshot{
		StateID: tokenBuffer.id,
		Type:    tokenBuffer.Type(),
		Schema:  "int64-le",
		Payload: payload,
	}, nil
}

func (tokenBuffer *TokenBuffer) Restore(ctx context.Context, snapshot Snapshot) error {
	if snapshot.Schema != "int64-le" {
		return fmt.Errorf("token_buffer: unsupported snapshot schema %q", snapshot.Schema)
	}

	if len(snapshot.Payload)%8 != 0 {
		return fmt.Errorf("token_buffer: payload length %d is not a multiple of 8", len(snapshot.Payload))
	}

	count := len(snapshot.Payload) / 8
	tokens := make([]int, count)

	for index := range tokens {
		tokens[index] = int(int64(binary.LittleEndian.Uint64(snapshot.Payload[index*8:])))
	}

	tokenBuffer.mu.Lock()
	tokenBuffer.tokens = tokens
	tokenBuffer.mu.Unlock()

	return nil
}

func (tokenBuffer *TokenBuffer) Inspect(ctx context.Context) (Inspection, error) {
	return Inspection{
		StateID: tokenBuffer.id,
		Type:    tokenBuffer.Type(),
		Values: map[string]any{
			"length": tokenBuffer.Length(),
		},
	}, nil
}
