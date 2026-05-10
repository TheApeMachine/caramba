/*
Package positional defines CPU positional-encoding operations as operation.Operation implementations.

Import as:

	import "github.com/theapemachine/caramba/backend/compute/cpu/operation/positional"

RoPE expects shape=[batch, num_heads, seq_len, head_dim] and data[0]=input tensor.
ALiBi expects shape=[num_heads, seq_len_q, seq_len_k] and returns the bias tensor.
*/
package positional
