package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/examples"

	_ "github.com/containerd/containerd/v2/cmd/containerd/builtins"
)

type Example interface {
	Run() error
}

var (
	user string

	exampleCmd = &cobra.Command{
		Use:   "example",
		Short: "Run Caramba examples",
		Long:  longExample,
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			switch args[0] {
			case "chat":
				return examples.NewChatExample(user).Run()
			default:
				return fmt.Errorf("unknown example: %s", args[0])
			}
		},
	}
)

func init() {
	exampleCmd.Flags().StringVarP(
		&user, "user", "u", "user", "The user name for the chat example",
	)

	rootCmd.AddCommand(exampleCmd)
}

var longExample = `
Run a caramba example.
`
