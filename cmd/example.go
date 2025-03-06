package cmd

import (
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/charmbracelet/log"
	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/workflow"
)

type ErrorUnknownExample struct {
	ExampleType string
}

func (e *ErrorUnknownExample) Error() string {
	return fmt.Sprintf("unknown example type: %s", e.ExampleType)
}

// Example command variables
var (
	exampleCmd = &cobra.Command{
		Use:   "example",
		Short: "Run example scenarios",
		Long:  longExample,
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			errnie.SetLevel(log.DebugLevel)
			log.Info("Starting example")

			// Initial message
			msg := &core.Message{Role: "user", Name: "Danny", Content: "Hi there!"}

			// Create shared streams between components
			msgToAgentStream := workflow.NewPipeStream()
			agentToProviderStream := workflow.NewPipeStream()
			providerToOutputStream := workflow.NewPipeStream()

			// Setup codecs explicitly with shared streams
			msgCodec := workflow.NewJSONCodec(msgToAgentStream)
			agentComponent := workflow.NewAgentComponent(
				ai.NewAgent(), msgToAgentStream, agentToProviderStream,
			)
			providerComponent := workflow.NewProviderComponent(
				provider.NewOpenAIProvider("", ""),
				agentToProviderStream,
				providerToOutputStream,
			)

			// Setup pipeline explicitly
			pipeline := workflow.NewPipeline(
				workflow.NewCodecStream(msgCodec),
				agentComponent,
				providerComponent,
			)

			var wg sync.WaitGroup
			wg.Add(1)

			// Handle final output
			go func() {
				defer wg.Done()
				defer pipeline.Close()

				io.Copy(os.Stdout, pipeline)
			}()

			go func() {
				if err := msgCodec.Write(msg); err != nil {
					log.Fatal("failed writing message:", err)
				}
			}()

			wg.Wait()

			return nil
		},
	}
)

func init() {
	rootCmd.AddCommand(exampleCmd)
}

var longExample = `
Example demonstrates various capabilities of the Caramba framework.
This command is primarily for testing and demonstration purposes.
`
