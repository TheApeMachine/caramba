package main

import (
	"github.com/theapemachine/caramba/cmd"
	"github.com/theapemachine/caramba/pkg/errnie"
)

func main() {
	if err := cmd.Execute(); err != nil {
		errnie.Error(err)
	}
}
