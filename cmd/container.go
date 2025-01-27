/*
Copyright © 2025 NAME HERE <EMAIL ADDRESS>
*/
package cmd

import (
	"fmt"
	"io"
	"regexp"
	"strings"

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
		promptRegex := regexp.MustCompile(`user@[^:]+:[^$]+\$`)
		ready := make(chan bool)

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
					if promptRegex.MatchString(strings.TrimSpace(string(buf[:n]))) {
						ready <- true
					}
				}
			}
		}()

		err := container.Start()
		if err != nil {
			fmt.Printf("Error starting container: %v\n", err)
			return
		}

		go func() {
			for {
				select {
				case <-ready:
					container.Conn.Write([]byte("echo hello\n"))
					return
				}
			}
		}()

		select {}
	},
}

func init() {
	rootCmd.AddCommand(containerCmd)
}
