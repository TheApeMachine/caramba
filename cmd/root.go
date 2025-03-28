/*
Package cmd implements the command-line interface for the Caramba framework.
It provides various commands for managing agents, running examples, and testing functionality.
*/
package cmd

import (
	"bytes"
	"embed"
	"fmt"
	"io"
	stdfs "io/fs"
	"log"
	"os"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/theapemachine/amsh/utils"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/fs"
)

/*
Embed a mini filesystem into the binary to hold the default config file.
This will be written to the home directory of the user running the service,
which allows a developer to easily override the config file.
*/
//go:embed cfg/*
var embedded embed.FS

//go:embed manifests/*
var manifests embed.FS

//go:embed scripts/*
var browserScripts embed.FS

/*
rootCmd represents the base command when called without any subcommands
*/
var (
	projectName  = "caramba"
	cfgFile      string
	openaiAPIKey string

	rootCmd = &cobra.Command{
		Use:   "caramba",
		Short: "A sophisticated multi-agent AI orchestration system",
		Long:  longRoot,
	}
)

/*
Execute is the main entry point for the Caramba CLI. It initializes the root command
and executes it.
*/
func Execute() error {
	errnie.Debug("Execute")
	return rootCmd.Execute()
}

/*
init is a function that initializes the root command and sets up the persistent flags.
*/
func init() {
	fmt.Println("cmd.root.init")

	cobra.OnInitialize(initConfig)

	rootCmd.PersistentFlags().StringVar(
		&cfgFile, "config", "config.yml", "config file (default is $HOME/."+projectName+"/config.yml)",
	)
	rootCmd.PersistentFlags().StringVar(
		&openaiAPIKey, "openai-api-key", "", "API key for the OpenAI provider",
	)

	// Initialize the filesystem store and load manifests
	if err := fs.NewStore().Initialize(manifests, "manifests"); err != nil {
		errnie.Error(err)
	}

	// Initialize the browser scripts
	if err := fs.NewStore().Initialize(browserScripts, "scripts"); err != nil {
		errnie.Error(err)
	}
}

/*
initConfig is a function that initializes the configuration for the Caramba CLI.
It writes the default config file to the user's home directory if it doesn't exist,
and then reads the config file from the user's home directory.
*/
func initConfig() {
	var err error

	errnie.Debug("initConfig")

	if err = writeConfig(); err != nil {
		log.Fatal(err)
	}

	viper.SetConfigName("config")
	viper.SetConfigType("yml")
	viper.AddConfigPath("$HOME/." + projectName)

	if err = viper.ReadInConfig(); err != nil {
		log.Fatal(err)
		return
	}
}

/*
writeConfig is a function that writes the default config file to the user's home directory.
*/
func writeConfig() (err error) {
	var (
		fh  stdfs.File
		buf bytes.Buffer
	)

	errnie.Debug("writeConfig")

	home, err := os.UserHomeDir()
	if errnie.Error(err) != nil {
		return err
	}

	fullPath := home + "/." + projectName + "/" + cfgFile

	if utils.CheckFileExists(fullPath) {
		return
	}

	if fh, err = embedded.Open("cfg/" + cfgFile); err != nil {
		return fmt.Errorf("failed to open embedded config file: %w", err)
	}

	defer fh.Close()

	if _, err = io.Copy(&buf, fh); err != nil {
		return fmt.Errorf("failed to read embedded config file: %w", err)
	}

	if err = os.MkdirAll(home+"/."+projectName, os.ModePerm); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}

	if err = os.WriteFile(fullPath, buf.Bytes(), 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	errnie.Debug("writeConfig", "fullPath", fullPath)

	return
}

/*
longRoot contains the detailed help text for the root command.
*/
var longRoot = `
Caramba is a powerful multi-agent AI orchestration system built in Go, designed to 
coordinate multiple AI providers and facilitate complex reasoning tasks through a 
pipeline-based architecture.

Key Features:
  - Multi-provider intelligence with smart load balancing
  - Graph-based pipeline architecture
  - Extensive tool system with browser, container, and database support
  - Advanced context management with 128k context window
  - Thread-safe operations and error handling

Examples:
  caramba example research - Run the research pipeline example
`
