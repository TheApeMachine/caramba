package prompt

import (
	"strings"

	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

type FragmentBuilder struct {
	Fragment *Fragment
}

type FragmentOption func(*FragmentBuilder)

func NewFragmentBuilder(opts ...FragmentOption) *FragmentBuilder {
	var (
		arena = capnp.SingleSegment(nil)
		seg   *capnp.Segment
		frag  Fragment
		err   error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return nil
	}

	if frag, err = NewRootFragment(seg); errnie.Error(err) != nil {
		return nil
	}

	fragBuilder := &FragmentBuilder{
		Fragment: &frag,
	}

	for _, opt := range opts {
		opt(fragBuilder)
	}

	return fragBuilder
}

func WithRole(role string) FragmentOption {
	return func(fragBuilder *FragmentBuilder) {
		fragBuilder.Fragment.SetTemplate(
			tweaker.WithVariable("prompts.system.role", "role", role),
		)
	}
}

func WithSkill(skill string) FragmentOption {
	return func(fragBuilder *FragmentBuilder) {
		fragBuilder.Fragment.SetTemplate(
			tweaker.WithVariable("prompts.system.skills", "skill", skill),
		)
	}
}

func WithResponsibility(responsibility string) FragmentOption {
	return func(fragBuilder *FragmentBuilder) {
		fragBuilder.Fragment.SetTemplate(
			tweaker.WithVariable("prompts.system.responsibilities", "responsibility", responsibility),
		)
	}
}

func WithIdentity(id string, name string) FragmentOption {
	return func(fragBuilder *FragmentBuilder) {
		fragBuilder.Fragment.SetTemplate(
			tweaker.WithVariables("prompts.system.identity", "name", name, "id", id),
		)
	}
}

func WithBuiltin(template string) FragmentOption {
	return func(fragBuilder *FragmentBuilder) {
		var frags strings.Builder

		frags.WriteString(tweaker.WithVariables(
			"prompts.system.role", "role", template, "domains", strings.Join(tweaker.GetStringSlice("prompts.builtin.roles."+template+".domains"), ", "),
		) + "\n\n")

		frags.WriteString(tweaker.WithVariable(
			"prompts.system.skills",
			"skills",
			strings.Join(tweaker.GetStringSlice("prompts.builtin.roles."+template+".skills"), ", "),
		) + "\n\n")

		frags.WriteString(tweaker.WithVariable(
			"prompts.system.responsibilities",
			"responsibilities",
			strings.Join(tweaker.GetStringSlice("prompts.builtin.roles."+template+".responsibilities"), ", "),
		) + "\n\n")

		fragBuilder.Fragment.SetTemplate(frags.String())
	}
}

func WithTemplate(template string) FragmentOption {
	return func(fragBuilder *FragmentBuilder) {
		fragBuilder.Fragment.SetTemplate(
			tweaker.Get("prompts.fragments."+template, ""),
		)
	}
}

func WithVariables(variables []string) FragmentOption {
	return func(fragBuilder *FragmentBuilder) {
		vrbls, err := fragBuilder.Fragment.Variables()
		if errnie.Error(err) != nil {
			return
		}

		nl, err := capnp.NewTextList(
			fragBuilder.Fragment.Segment(), int32(vrbls.Len()+len(variables)),
		)
		if errnie.Error(err) != nil {
			return
		}

		for i := 0; i < vrbls.Len(); i++ {
			val, err := vrbls.At(i)
			if errnie.Error(err) != nil {
				continue
			}
			nl.Set(i, val)
		}

		for i, variable := range variables {
			nl.Set(i+vrbls.Len(), variable)
		}

		fragBuilder.Fragment.SetVariables(nl)
	}
}
