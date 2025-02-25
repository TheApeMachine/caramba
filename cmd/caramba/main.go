package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/agent/examples"
	"github.com/theapemachine/errnie"
)

func initConfig() {
	// Initialize Viper configuration
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath(".")
	viper.AddConfigPath("$HOME/.caramba")
	viper.AddConfigPath("/etc/caramba")

	// Read the config file
	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			fmt.Printf("Error reading config file: %v\n", err)
		}
		// Config file not found, using defaults
	}
}

func main() {
	// Initialize the configuration
	initConfig()

	// Create the root command
	rootCmd := &cobra.Command{
		Use:   "caramba",
		Short: "Caramba - AI agent framework",
		Long:  `Caramba is a framework for building AI agents with tools and workflows.`,
	}

	// Setup the browser example command
	browserCmd := &cobra.Command{
		Use:   "browser",
		Short: "Run browser example",
		Long:  "Run an example using the browser tool",
		Run: func(cmd *cobra.Command, args []string) {
			apiKey, _ := cmd.Flags().GetString("api-key")
			url, _ := cmd.Flags().GetString("url")
			enableLogging, _ := cmd.Flags().GetBool("log")
			disableConsole, _ := cmd.Flags().GetBool("no-console")

			// Setup logging
			if enableLogging {
				logPath := viper.GetString("log_path")
				if logPath == "" {
					logPath = "logs/caramba.log"
				}
				// Configure logging
				os.Setenv("LOG_LEVEL", "debug")
				os.Setenv("LOGFILE", "true")
				if logPath != "" {
					os.Setenv("LOG_PATH", logPath)
				}
				if disableConsole {
					os.Setenv("CONSOLE", "false")
				} else {
					os.Setenv("CONSOLE", "true")
				}
				errnie.InitLogger()
				fmt.Println("Logging initialized at:", logPath)
			}

			// Run the browser example
			if err := examples.BrowserExample(apiKey, url); err != nil {
				fmt.Printf("Error running browser example: %v\n", err)
				os.Exit(1)
			}
		},
	}
	browserCmd.Flags().String("api-key", "", "OpenAI API key")
	browserCmd.Flags().String("url", "", "URL to browse (default: Hacker News)")
	browserCmd.Flags().Bool("log", false, "Enable logging")
	browserCmd.Flags().Bool("no-console", false, "Disable console output")
	rootCmd.AddCommand(browserCmd)

	// Setup the multi-provider browser test command
	multiProviderCmd := &cobra.Command{
		Use:   "test-providers",
		Short: "Test browser with multiple providers",
		Long:  "Run browser examples with both OpenAI and Anthropic providers",
		Run: func(cmd *cobra.Command, args []string) {
			openaiKey, _ := cmd.Flags().GetString("openai-key")
			anthropicKey, _ := cmd.Flags().GetString("anthropic-key")
			url, _ := cmd.Flags().GetString("url")
			enableLogging, _ := cmd.Flags().GetBool("log")
			disableConsole, _ := cmd.Flags().GetBool("no-console")

			// Setup logging
			if enableLogging {
				logPath := viper.GetString("log_path")
				if logPath == "" {
					logPath = "logs/caramba.log"
				}
				// Configure logging
				os.Setenv("LOG_LEVEL", "debug")
				os.Setenv("LOGFILE", "true")
				if logPath != "" {
					os.Setenv("LOG_PATH", logPath)
				}
				if disableConsole {
					os.Setenv("CONSOLE", "false")
				} else {
					os.Setenv("CONSOLE", "true")
				}
				errnie.InitLogger()
				fmt.Println("Logging initialized at:", logPath)
			}

			// Check if at least one API key is provided
			if openaiKey == "" && anthropicKey == "" {
				fmt.Println("Error: You must provide at least one API key (OpenAI or Anthropic)")
				os.Exit(1)
			}

			// Run the multi-provider browser test
			if err := examples.TestMultiProviderBrowser(openaiKey, anthropicKey, url); err != nil {
				fmt.Printf("Error running multi-provider test: %v\n", err)
				os.Exit(1)
			}
		},
	}
	multiProviderCmd.Flags().String("openai-key", "", "OpenAI API key")
	multiProviderCmd.Flags().String("anthropic-key", "", "Anthropic API key")
	multiProviderCmd.Flags().String("url", "", "URL to browse (default: Hacker News)")
	multiProviderCmd.Flags().Bool("log", false, "Enable logging")
	multiProviderCmd.Flags().Bool("no-console", false, "Disable console output")
	rootCmd.AddCommand(multiProviderCmd)

	// Execute the root command
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
