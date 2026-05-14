package xla

/*
OperationRegistry constructs XLA-backed compute operations.
The build-tag-specific implementation wires methods to PJRT-backed kernels.
*/
type OperationRegistry struct{}

/*
NewOperationRegistry instantiates the XLA operation registry.
*/
func NewOperationRegistry() *OperationRegistry {
	return &OperationRegistry{}
}
