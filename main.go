/*
Package main is the entry point for the Caramba agent framework.

Caramba is a specialized agent framework in Go that provides a robust architecture
for creating, managing, and orchestrating AI agents with various capabilities including
multi-provider intelligence, iteration, communication, memory management, and tool integration.

The main package initializes the command-line interface and executes the root command
which provides access to all Caramba functionality.
*/
package main

import "github.com/theapemachine/caramba/cmd"

/*
main is the entry point function for the Caramba application.
It calls the Execute function from the cmd package to start the CLI.
*/
func main() {
	cmd.Execute()
}
