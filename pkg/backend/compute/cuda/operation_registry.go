package cuda

/*
OperationRegistry constructs CUDA-backed compute operations.
The concrete constructor methods live in build-tag-specific files.
*/
type OperationRegistry struct{}

/*
NewOperationRegistry instantiates the CUDA operation registry.
*/
func NewOperationRegistry() *OperationRegistry {
	return &OperationRegistry{}
}
