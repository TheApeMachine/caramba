/*
Copyright © 2024 NAME HERE <EMAIL ADDRESS>
*/
package cmd

import (
	"bytes"
	"embed"
	"fmt"
	"io"
	"io/fs"
	"log"
	"os"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/theapemachine/amsh/utils"
	"github.com/theapemachine/errnie"
)

/*
Embed a mini filesystem into the binary to hold the default config file.
This will be written to the home directory of the user running the service,
which allows a developer to easily override the config file.
*/
//go:embed cfg/*
var embedded embed.FS

/*
rootCmd represents the base command when called without any subcommands
*/
var (
	projectName = "caramba"
	cfgFile     string

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
func Execute() {
	err := rootCmd.Execute()
	if err != nil {
		os.Exit(1)
	}
}

/*
init is a function that initializes the root command and sets up the persistent flags.
*/
func init() {
	cobra.OnInitialize(initConfig)

	rootCmd.PersistentFlags().StringVar(
		&cfgFile, "config", "config.yml", "config file (default is $HOME/."+projectName+"/config.yml)",
	)
	rootCmd.Flags().BoolP("toggle", "t", false, "Help message for toggle")
}

/*
initConfig is a function that initializes the configuration for the Caramba CLI.
It writes the default config file to the user's home directory if it doesn't exist,
and then reads the config file from the user's home directory.
*/
func initConfig() {
	var err error

	if err = writeConfig(); err != nil {
		errnie.Error(err)
		log.Fatal(err)
	}

	viper.SetConfigName("config")
	viper.SetConfigType("yml")
	viper.AddConfigPath("$HOME/." + projectName)

	if err = viper.ReadInConfig(); err != nil {
		errnie.Error(err)
		log.Println("failed to read config file", err)
		return
	}
}

/*
writeConfig is a function that writes the default config file to the user's home directory.
*/
func writeConfig() (err error) {
	var (
		home, _ = os.UserHomeDir()
		fh      fs.File
		buf     bytes.Buffer
	)

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

	if err = os.Mkdir(home+"/."+projectName, os.ModePerm); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}

	if err = os.WriteFile(fullPath, buf.Bytes(), 0644); err != nil {
		errnie.Error(err)
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return
}

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
  caramba example dev      - Run the development pipeline example
  caramba example chat     - Run the simple chat example
  caramba test             - Run the test setup
`
