package manifest

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation"
)

/*
Constructor builds an Operation from a config map produced by the manifest parser.
Each operation package registers itself via init().
*/
type Constructor func(config map[string]any) (operation.Operation, error)

/*
OperationRegistry maps operation identifiers to constructors.
*/
type OperationRegistry struct {
	constructors map[string]Constructor
}

/*
globalRegistry is the default registry used by Compiler and package-level helpers.
*/
var globalRegistry = NewOperationRegistry()

/*
NewOperationRegistry constructs an empty OperationRegistry.
Used for isolated registries in tests or custom Compiler wiring.
*/
func NewOperationRegistry() *OperationRegistry {
	return &OperationRegistry{
		constructors: make(map[string]Constructor),
	}
}

/*
Register binds constructor on the global registry.
Call from init() in each operation package.
*/
func Register(operationID string, constructor Constructor) {
	globalRegistry.Register(operationID, constructor)
}

/*
Register binds constructor to operationID within this OperationRegistry.
*/
func (operationRegistry *OperationRegistry) Register(operationID string, constructor Constructor) {
	operationRegistry.constructors[operationID] = constructor
}

/*
Build looks up constructor for operationID on the global registry and instantiates it.
*/
func Build(operationID string, config map[string]any) (operation.Operation, error) {
	return globalRegistry.Build(operationID, config)
}

/*
Build looks up constructor for operationID on this OperationRegistry.
*/
func (operationRegistry *OperationRegistry) Build(operationID string, config map[string]any) (operation.Operation, error) {
	constructor, ok := operationRegistry.constructors[operationID]

	if !ok {
		return nil, fmt.Errorf("manifest: unknown operation %q", operationID)
	}

	return constructor(config)
}

/*
Registered returns all identifiers on the global registry for tooling.
*/
func Registered() []string {
	return globalRegistry.Registered()
}

/*
Registered returns identifiers registered with this OperationRegistry.
*/
func (operationRegistry *OperationRegistry) Registered() []string {
	operationIDs := make([]string, 0, len(operationRegistry.constructors))

	for identifier := range operationRegistry.constructors {
		operationIDs = append(operationIDs, identifier)
	}

	return operationIDs
}
