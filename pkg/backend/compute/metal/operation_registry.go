package metal

/*
OperationRegistry constructs Metal-backed compute operations.
Build-tag-specific files provide the native and unavailable method sets.
*/
type OperationRegistry struct{}

/*
NewOperationRegistry instantiates the Metal operation registry.
*/
func NewOperationRegistry() *OperationRegistry {
	return &OperationRegistry{}
}
