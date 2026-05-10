/*
Package memory provides associative and dynamic memory operations for exotic neural architectures.

These are stateful operations: patterns are stored in the struct and recalled via Forward.
Two complementary ops per memory type handle the store/recall split:

  - hopfield.store  — writes patterns into a HopfieldMemory
  - hopfield.recall — retrieves the nearest stored pattern via synchronous updates

Import as:

	import _ "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/memory"

The blank import triggers init() which registers all memory ops with the global manifest registry.
*/
package memory
