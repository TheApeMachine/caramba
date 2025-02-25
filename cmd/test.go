package cmd

import (
	"context"
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/caramba/pkg/agent/llm"
	"github.com/theapemachine/caramba/pkg/agent/memory"
	"github.com/theapemachine/caramba/pkg/agent/tools"
	"github.com/theapemachine/caramba/pkg/agent/workflow"
	"github.com/theapemachine/errnie"
)

/*
testCmd is a command that is used to test the agent framework.
*/
var testCmd = &cobra.Command{
	Use:   "test",
	Short: "Test the agent framework",
	Long:  `Executes a test setup to demonstrate the agent framework capabilities`,
	Run: func(cmd *cobra.Command, args []string) {
		os.Setenv("LOG_LEVEL", "debug")
		os.Setenv("LOGFILE", "true")
		errnie.InitLogger()

		errnie.Info("🌈 Creating test agent")

		// Get API key from flags or environment
		apiKey, _ := cmd.Flags().GetString("api-key")
		if apiKey == "" {
			apiKey = os.Getenv("OPENAI_API_KEY")
		}

		if apiKey == "" {
			errnie.Error(fmt.Errorf("API key not provided. Use --api-key flag or set OPENAI_API_KEY environment variable"))
			return
		}

		// Create an agent using the builder pattern
		agent := core.NewAgentBuilder("TestAgent").
			WithLLM(llm.NewOpenAIProvider(apiKey, "gpt-4o-mini")).
			WithMemory(memory.NewInMemoryStore()).
			WithTool(tools.NewCalculator()).
			Build()

		// Test a simple query
		errnie.Info("💬 Testing simple query")
		resp, err := agent.Execute(context.Background(), "What is 2+2?")
		if err != nil {
			errnie.Error(err)
			return
		}
		fmt.Println("Agent Response:")
		fmt.Println(resp)

		// Test a workflow
		errnie.Info("🔄 Testing workflow")
		wf := workflow.NewWorkflow().
			AddStep("calculate", tools.NewCalculator(), map[string]interface{}{
				"expression": "3 * 4 + 5",
			}).
			AddStep("format_result", tools.NewFormatter(), map[string]interface{}{
				"template": "The result of the calculation is {{.calculate}}",
			})

		results, err := agent.RunWorkflow(context.Background(), wf, nil)
		if err != nil {
			errnie.Error(err)
			return
		}

		fmt.Println("Workflow Results:")
		for k, v := range results {
			fmt.Printf("%s: %v\n", k, v)
		}
	},
}

func init() {
	rootCmd.AddCommand(testCmd)
	testCmd.Flags().String("api-key", "", "API key for the LLM provider (or set OPENAI_API_KEY env var)")
}
