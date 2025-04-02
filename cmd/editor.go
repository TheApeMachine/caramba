package cmd

import (
	_ "github.com/containerd/containerd/v2/cmd/containerd/builtins"
	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/piaf"
)

var (
	editorCmd = &cobra.Command{
		Use:   "editor",
		Short: "Run editor",
		Long:  longEditor,
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			editor := piaf.NewEditor()
			terminal := piaf.NewTerminal(editor)
			return terminal.Run()
		},
	}
)

func init() {
	rootCmd.AddCommand(editorCmd)
}

var longEditor = `
Editor is a simple text editor that allows you to edit text in a terminal.
`
