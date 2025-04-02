package core

import (
	"strings"

	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/utils"
)

type PromptBuilder struct {
	*Prompt
}

type PromptOption func(*PromptBuilder)

func NewPromptBuilder(options ...PromptOption) *PromptBuilder {
	errnie.Debug("ai.NewPromptBuilder")

	var (
		cpnp   = utils.NewCapnp()
		prompt Prompt
		err    error
	)

	if prompt, err = NewRootPrompt(cpnp.Seg); errnie.Error(err) != nil {
		return nil
	}

	return &PromptBuilder{
		Prompt: &prompt,
	}
}

func (builder *PromptBuilder) String() string {
	sb := strings.Builder{}

	fragments, err := builder.Fragments()

	if errnie.Error(err) != nil {
		return ""
	}

	for i := range fragments.Len() {
		fragment := fragments.At(i)

		template, err := fragment.Template()

		if errnie.Error(err) != nil {
			return ""
		}

		variables, err := fragment.Variables()

		if errnie.Error(err) != nil {
			return ""
		}

		for j := range variables.Len() {
			variable, err := variables.At(j)

			if errnie.Error(err) != nil {
				return ""
			}

			template = strings.Replace(
				template, "{{"+variable+"}}", variable, 1,
			)
		}

		sb.WriteString(template + "\n")
	}

	return sb.String()
}

func WithFragment(fragment string, variables ...string) PromptOption {
	return func(builder *PromptBuilder) {
		fl, err := NewFragment_List(builder.Segment(), int32(1))

		if errnie.Error(err) != nil {
			return
		}

		frgmnt, err := NewFragment(builder.Segment())

		if errnie.Error(err) != nil {
			return
		}

		if err = frgmnt.SetTemplate(fragment); errnie.Error(err) != nil {
			return
		}

		vrbls, err := capnp.NewTextList(builder.Segment(), int32(len(variables)))

		if errnie.Error(err) != nil {
			return
		}

		for i, variable := range variables {
			vrbls.Set(i, variable)
		}

		if err = frgmnt.SetVariables(vrbls); errnie.Error(err) != nil {
			return
		}

		if err = fl.Set(0, frgmnt); errnie.Error(err) != nil {
			return
		}

		if err = builder.SetFragments(fl); errnie.Error(err) != nil {
			return
		}
	}
}
