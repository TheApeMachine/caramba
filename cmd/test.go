package cmd

import (
	"context"
	"encoding/json"
	"os"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/datura"
	"github.com/theapemachine/caramba/environment"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/system"
	"github.com/theapemachine/caramba/tools"
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

		agent := ai.NewAgent(
			ai.NewIdentity("ui"),
			[]provider.Tool{
				tools.NewMessageTool().Convert(),
				tools.NewAgentTool().Convert(),
				tools.NewCompletionTool().Convert(),
			},
		)

		queue := system.NewQueue()
		queue.AddAgent(agent)

		executor := environment.NewExecutor(agent)
		executor.Run(context.Background())

		artifact := datura.NewArtifactBuilder(
			datura.MediaTypeTextPlain,
			datura.ArtifactRoleUser,
			datura.ArtifactScopePrompt,
		)

		msg := tools.MessageParams{
			To:       "ui",
			From:     "user",
			Topic:    "test",
			Content:  "I would like to explore new machine learning architectures, designed to be trainable on consumer grade hardware.",
			Priority: 0,
		}

		payload, err := json.Marshal(msg)

		if errnie.Error(err) != nil {
			return
		}

		artifact.SetPayload(payload)
		message, err := artifact.Build()

		if errnie.Error(err) != nil {
			return
		}

		queue.SendMessage(message)

		select {}
	},
}

func init() {
	rootCmd.AddCommand(testCmd)
}
