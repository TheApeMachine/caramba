package tokenizer

import (
	"errors"
	"strings"
)

/*
Tokenizer primitives — BPE encode and decode plus the vocabulary
load helper. These run on the host side; they don't go through SIMD
or device dispatch.

The host kernel registry exposes encode/decode as Run entries that
operate on byte/int32 tensors. The actual vocabulary / merges live
in a Tokenizer struct that callers construct from a model bundle.
*/

type Tokenizer struct {
	Vocab    map[string]int32
	Inverse  map[int32]string
	Merges   [][2]string
	UNK      int32
	BOS      int32
	EOS      int32
	PadToken int32
}

var ErrTokenizerNotConfigured = errors.New("tokenizer: vocabulary not configured")

func NewTokenizer() *Tokenizer {
	return &Tokenizer{
		Vocab:   map[string]int32{},
		Inverse: map[int32]string{},
	}
}

/*
Encode applies a greedy longest-match BPE encoding over the input
runes. Returns a slice of token IDs.
*/
func (tok *Tokenizer) Encode(text string) ([]int32, error) {
	if len(tok.Vocab) == 0 {
		return nil, ErrTokenizerNotConfigured
	}

	tokens := []int32{}
	cursor := 0

	for cursor < len(text) {
		matchEnd := cursor + 1

		for candidateEnd := len(text); candidateEnd > cursor; candidateEnd-- {
			candidate := text[cursor:candidateEnd]

			if id, ok := tok.Vocab[candidate]; ok {
				tokens = append(tokens, id)
				matchEnd = candidateEnd
				break
			}
		}

		if matchEnd == cursor+1 {
			tokens = append(tokens, tok.UNK)
		}

		cursor = matchEnd
	}

	return tokens, nil
}

/*
Decode reverses Encode by joining each token's surface form. Tokens
outside the vocabulary fall back to the empty string.
*/
func (tok *Tokenizer) Decode(tokens []int32) string {
	if len(tok.Inverse) == 0 {
		return ""
	}

	var builder strings.Builder

	for _, id := range tokens {
		if surface, ok := tok.Inverse[id]; ok {
			builder.WriteString(surface)
		}
	}

	return builder.String()
}
