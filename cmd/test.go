package cmd

import (
	"errors"
	"os"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/errnie"
)

/*
testCmd is a command that is used to test the agent.
*/
var testCmd = &cobra.Command{
	Use:   "test",
	Short: "Used to test the agent",
	Long:  `Executes a test setup`,
	Run: func(cmd *cobra.Command, args []string) {
		os.Setenv("LOG_LEVEL", "debug")
		os.Setenv("LOGFILE", "true")
		errnie.InitLogger()

		v := viper.GetViper()
		system := v.GetString("prompts.templates.systems.default")

		if system == "" {
			errnie.Error(errors.New("system is empty"))
			return
		}

		prvdr := provider.NewBalancedProvider()
		err := prvdr.Initialize(cmd.Context())
		if err != nil {
			errnie.Error(err)
			return
		}

		dctx := drknow.QuickContext(
			system,
			"codeswitch",
			"noexplain",
			"silentfail",
			"scratchpad",
		)

		agent := ai.NewAgent(
			dctx,
			prvdr,
			"reasoner",
			10,
		)

		stream.NewConsumer().Print(
			agent.Generate(
				cmd.Context(),
				provider.NewMessage(
					provider.RoleUser,
					"Find out about the latest scandals in the tech industry.",
				),
			),
			false,
		)
	},
}

func init() {
	rootCmd.AddCommand(testCmd)
}
