package main

import (
	"github.com/theapemachine/caramba/cmd"
)

func main() {
	if err := cmd.Execute(); err != nil {
		panic(err)
	}
}
