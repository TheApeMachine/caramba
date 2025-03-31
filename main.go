package main

import (
	clog "github.com/charmbracelet/log"
	"github.com/theapemachine/caramba/cmd"
	"github.com/theapemachine/caramba/pkg/errnie"
)

func main() {
	errnie.SetLevel(clog.InfoLevel)

	if err := cmd.Execute(); err != nil {
		errnie.Error(err)
	}
}
