// Package hawkes implements Hawkes-process and Markov-blanket CPU kernels.
package hawkes

import "github.com/theapemachine/caramba/pkg/backend/compute/kernels"

type (
	Kernel    = kernels.Kernel
	Signature = kernels.Signature
)

var Default = kernels.Default
