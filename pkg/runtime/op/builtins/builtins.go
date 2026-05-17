/*
Package builtins blank-imports every op subpackage so callers can
enable the full runtime op set with a single underscore import:

	import _ "github.com/theapemachine/caramba/pkg/runtime/op/builtins"

The executor consults op.Default during dispatch; each op subpackage
registers its ops onto op.Default in its init function.
*/
package builtins

import (
	_ "github.com/theapemachine/caramba/pkg/runtime/op/control"
	_ "github.com/theapemachine/caramba/pkg/runtime/op/graph"
	_ "github.com/theapemachine/caramba/pkg/runtime/op/io"
	_ "github.com/theapemachine/caramba/pkg/runtime/op/sampler"
	_ "github.com/theapemachine/caramba/pkg/runtime/op/scheduler"
	_ "github.com/theapemachine/caramba/pkg/runtime/op/stateop"
	_ "github.com/theapemachine/caramba/pkg/runtime/op/telemetryops"
	_ "github.com/theapemachine/caramba/pkg/runtime/op/tokenize"
	_ "github.com/theapemachine/caramba/pkg/runtime/op/value"
)
