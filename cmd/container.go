/*
Copyright © 2025 NAME HERE <EMAIL ADDRESS>
*/
package cmd

import (
	"fmt"
	"io"
	"time"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/tools"
)

// containerBridge implements io.ReadWriteCloser
type containerBridge struct {
	reader *io.PipeReader
	writer *io.PipeWriter
}

func (b *containerBridge) Read(p []byte) (n int, err error) {
	return b.reader.Read(p)
}

func (b *containerBridge) Write(p []byte) (n int, err error) {
	return b.writer.Write(p)
}

func (b *containerBridge) Close() error {
	b.reader.Close()
	return b.writer.Close()
}

// containerCmd represents the container command
var containerCmd = &cobra.Command{
	Use:   "container",
	Short: "Test container",
	Long:  `Test container`,
	Run: func(cmd *cobra.Command, args []string) {
		container := tools.NewContainer()

		// Now connect to container
		if err := container.Connect(cmd.Context(), nil); err != nil {
			fmt.Printf("Error connecting to container: %v\n", err)
			return
		}

		// Set up reader before connecting
		go func() {
			buf := make([]byte, 4096)
			for {
				n, err := container.Conn.Read(buf) // Read from the bridge instead
				if err != nil {
					if err != io.EOF {
						fmt.Printf("Error reading from container bridge: %v\n", err)
					}
					break
				}
				if n > 0 {
					fmt.Print(string(buf[:n])) // Print the output
				}
			}
		}()

		err := container.Start()
		if err != nil {
			fmt.Printf("Error starting container: %v\n", err)
			return
		}

		// Wait a bit for the container to initialize and show prompt
		time.Sleep(5 * time.Second)

		// Helper function to write commands and echo them
		writeCommand := func(cmd string) {
			fmt.Print(cmd) // Echo the command
			container.Conn.Write([]byte(cmd))
		}

		// Write test commands
		writeCommand("echo hello\n")
		time.Sleep(5 * time.Second)
		writeCommand("ls -la\n")
		time.Sleep(5 * time.Second)
		writeCommand("exit\n")
	},
}

func init() {
	rootCmd.AddCommand(containerCmd)
}
