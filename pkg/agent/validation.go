package agent

import (
	"github.com/cohesivestack/valgo"
)

type GenericValidator struct {
}

func NewGenericValidator() *GenericValidator {
	return &GenericValidator{}
}

func (validator *GenericValidator) Validate(out any) error {
	msg := out.(Message)

	return valgo.Is(
		valgo.String(msg.Role).Not().Blank(),
	).Error()
}
