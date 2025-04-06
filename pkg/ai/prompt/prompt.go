package prompt

import (
	"strings"

	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type PromptBuilder struct {
	Prompt *Prompt
}

type PromptOption func(*PromptBuilder)

func New(opts ...PromptOption) *PromptBuilder {
	var (
		arena  = capnp.SingleSegment(nil)
		seg    *capnp.Segment
		prompt Prompt
		err    error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return nil
	}

	if prompt, err = NewRootPrompt(seg); errnie.Error(err) != nil {
		return nil
	}

	promptBuilder := &PromptBuilder{
		Prompt: &prompt,
	}

	_, err = promptBuilder.Prompt.NewFragments(1)
	if errnie.Error(err) != nil {
		return nil
	}

	for _, opt := range opts {
		opt(promptBuilder)
	}

	return promptBuilder
}

func (prompt *PromptBuilder) Bytes() []byte {
	return []byte(prompt.String())
}

func (prompt *PromptBuilder) String() string {
	var frags strings.Builder

	fragments, err := prompt.Prompt.Fragments()

	if errnie.Error(err) != nil {
		return ""
	}

	for i := range fragments.Len() {
		template, err := fragments.At(i).Template()

		if errnie.Error(err) != nil {
			return ""
		}

		frags.WriteString(template + "\n\n")
	}

	return frags.String()
}

func WithFragments(fragments ...*FragmentBuilder) PromptOption {
	return func(pb *PromptBuilder) {
		frgmts, err := pb.Prompt.Fragments()
		if errnie.Error(err) != nil {
			return
		}

		frags, err := pb.Prompt.NewFragments(
			int32(frgmts.Len() + len(fragments)),
		)

		if errnie.Error(err) != nil {
			return
		}

		for i := range frgmts.Len() {
			val := frgmts.At(i)
			frags.Set(i, val)
		}

		for i, fragment := range fragments {
			if err := frags.Set(i, *fragment.Fragment); errnie.Error(err) != nil {
				continue
			}
		}
	}
}
