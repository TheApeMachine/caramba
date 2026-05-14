/*
Package shape defines CPU shape-manipulation operations as operation.Operation implementations.

Operations:
  - Transpose: swap two dimensions of an N-D tensor
  - Reshape: change shape metadata while keeping the data flat
  - ViewAsHeads: [B,T,D] -> [B,H,T,D/H] for multi-head attention
  - MergeHeads: [B,H,T,head_dim] -> [B,T,H*head_dim]
  - Concat: concatenate tensors along an axis
  - Split: split a tensor into equal-sized chunks along an axis

All operations use the universal signature:

	Forward(stateDict *state.Dict) (*state.Dict, error)

Inner copy loops use SIMD (AVX2 / SSE2 / NEON) for bulk memory movement.
*/
package shape
