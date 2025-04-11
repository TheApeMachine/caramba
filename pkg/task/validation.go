package task

type GenericValidator struct {
}

func NewGenericValidator() *GenericValidator {
	return &GenericValidator{}
}

// Validate currently performs no validation.
// TODO: Implement proper struct validation, potentially using tags on parameter structs
// and a library like go-playground/validator integrated with Fiber.
func (validator *GenericValidator) Validate(out any) error {
	// Placeholder: No validation performed currently.
	// The previous implementation incorrectly assumed 'out' was always task.Message
	// msg := out.(Message)
	// return valgo.Is(
	// 	valgo.String(msg.Role).Not().Blank(),
	// ).Error()
	return nil
}
