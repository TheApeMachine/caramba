package cmd

import (
	"os"

	"github.com/google/uuid"
	sdk "github.com/openai/openai-go"
	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/ai"
	"github.com/theapemachine/caramba/process/persona"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/system"
	"github.com/theapemachine/caramba/utils"
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

		agentProcess := &persona.Agent{}
		schema := agentProcess.GenerateSchema()

		agent := ai.NewAgent(uuid.New().String(), utils.NewName())
		agent.Stream()

		q := system.NewQueue()
		q.Start()

		q.Ingress(&system.Envelope{
			To: "",
			Payload: &provider.StructuredParams{
				Messages: []sdk.ChatCompletionMessageParamUnion{
					sdk.SystemMessage("Act like you are a highly efficient, advanced AI agent, part of a complex, sophisticated multi-agent system."),
					sdk.UserMessage(""),
				},
				Schema: sdk.ResponseFormatJSONSchemaJSONSchemaParam{
					Name:        sdk.F(agentProcess.Name()),
					Description: sdk.F(agentProcess.Description()),
					Schema:      sdk.F(any(schema)),
					Strict:      sdk.Bool(true),
				},
				Tools: []sdk.ChatCompletionToolParam{},
			},
		})

		select {}
	},
}

func init() {
	rootCmd.AddCommand(testCmd)
}
