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

		errnie.Info("🌈 "+"creating", "agent", "ui")

		agent := ai.NewAgent(
			ai.NewIdentity("ui"),
			[]provider.Tool{
				tools.NewCommandTool().Convert(),
				tools.NewCompletionTool().Convert(),
				tools.NewMessageTool().Convert(),
				tools.NewAgentTool().Convert(),
			},
		)

		errnie.Info("✨ "+agent.Identity.Name, "queue", "add")

		queue := system.NewQueue()
		queue.AddAgent(agent)

		pool := environment.NewPool()

		errnie.Info("💫 "+agent.Identity.Name, "executor", "add")

		executor := environment.NewExecutor(agent)
		pool.AddExecutor(executor)

		executor.Run(context.Background())
		errnie.Info("🚀 "+agent.Identity.Name, "executor", "running")

		artifact := datura.NewArtifactBuilder(
			datura.MediaTypeTextPlain,
			datura.ArtifactRoleBroadcast,
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

		errnie.Info("📭 "+agent.Identity.Name, "queue", "send")

		queue.SendMessage(message)

		errnie.Info("📬 "+agent.Identity.Name, "queue", "sent")

		select {}
	},
}

func init() {
	rootCmd.AddCommand(testCmd)
}
