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

// rootCmd represents the base command when called without any subcommands
var (
	projectName = "caramba"
	cfgFile     string

	rootCmd = &cobra.Command{
		Use:   "caramba",
		Short: "A brief description of your application",
		Long: `A longer description that spans multiple lines and likely contains
examples and usage of using your application. For example:

Cobra is a CLI library for Go that empowers applications.
This application is a tool to generate the needed files
to quickly create a Cobra application.`,
		// Uncomment the following line if your bare application
		// has an action associated with it:
		// Run: func(cmd *cobra.Command, args []string) { },
	}
)

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	err := rootCmd.Execute()
	if err != nil {
		os.Exit(1)
	}
}

func init() {
	cobra.OnInitialize(initConfig)

	rootCmd.PersistentFlags().StringVar(
		&cfgFile, "config", "config.yml", "config file (default is $HOME/."+projectName+"/config.yml)",
	)
	rootCmd.Flags().BoolP("toggle", "t", false, "Help message for toggle")
}

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
