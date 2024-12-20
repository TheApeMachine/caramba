package ai

import (
	"github.com/theapemachine/caramba/process/fractal"
	"github.com/theapemachine/caramba/process/holographic"
	"github.com/theapemachine/caramba/process/quantum"
	"github.com/theapemachine/caramba/process/temporal"
)

type Process interface {
	GenerateSchema() string
}

var processMap = map[string]Process{
	"temporal":    &temporal.Process{},
	"holographic": &holographic.Process{},
	"quantum":     &quantum.Process{},
	"fractal":     &fractal.Process{},
}
