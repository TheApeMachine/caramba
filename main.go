package main

import (
	"github.com/theapemachine/caramba/cmd"
	"github.com/theapemachine/caramba/pkg/errnie"
)

func main() {
	errnie.SetLevel(errnie.TraceLevel)

	if err := cmd.Execute(); err != nil {
		errnie.Error(err)
	}
}
