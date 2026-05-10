package operation

import cpuop "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation"

/*
Operation is an alias of the CPU operation interface so callers may import either
github.com/theapemachine/caramba/pkg/backend/compute/operation or
.../cpu/operation without divergent definitions.
*/
type Operation = cpuop.Operation
