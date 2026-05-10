package cmd

import (
	"embed"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

/*
Embed a mini filesystem into the binary to hold the default config file.
This will be written to the home directory of the user running the service,
which allows a developer to easily override the config file.
*/
//go:embed asset/config.yml
var embedded embed.FS

/*
rootCmd represents the base command when called without any subcommands
*/
var (
	cfgFile string

	rootCmd = &cobra.Command{
		Use:   "caramba",
		Short: "Caramba is a fully featured machine learning research platform.",
		Long:  rootLong,
		// Run: func(cmd *cobra.Command, args []string) {
		// 	fmt.Println("Hello, World!")
		// },
	}
)

/*
Execute adds all child commands to the root command and sets flags appropriately.
This is called by main.main(). It only needs to happen once to the rootCmd.
*/
func Execute() {
	err := rootCmd.Execute()

	if err != nil {
		os.Exit(1)
	}
}

func init() {
	cobra.OnInitialize(initConfig)

	rootCmd.PersistentFlags().StringVar(
		&cfgFile,
		"config",
		"",
		"path to config file (default: try cmd/cfg/config.yml, ./config.yml, $HOME/.mosaic/config.yml, then embedded default)",
	)
}

/*
initConfig loads config.yml from, in order:
  - path given by --config (if set)
  - ./cmd/cfg/config.yml (repo checkout)
  - ./config.yml
  - $HOME/.six/config.yml
  - embedded cmd/cfg/config.yml
*/
func initConfig() {
	viper.SetConfigType("yml")

	tryRead := func(path string) error {
		viper.SetConfigFile(path)
		return viper.ReadInConfig()
	}

	loaded := false

	if rootCmd.PersistentFlags().Changed("config") && strings.TrimSpace(cfgFile) != "" {
		if err := tryRead(cfgFile); err == nil {
			loaded = true
		} else {
			fmt.Fprintf(os.Stderr, "mosaic: config file %q: %v\n", cfgFile, err)

			os.Exit(1)
		}
	}

	if !loaded {
		paths := []string{
			"cmd/cfg/config.yml",
			"config.yml",
		}

		if home, err := os.UserHomeDir(); err == nil {
			paths = append(paths, filepath.Join(home, ".mosaic", "config.yml"))
		}

		for _, p := range paths {
			if err := tryRead(p); err == nil {
				loaded = true
				break
			}
		}
	}

	if !loaded {
		fmt.Println("config file not found, using embedded config")
		cfgReader, openErr := embedded.Open("asset/config.yml")

		if openErr != nil {
			fmt.Printf("embedded config file not found: %v\n", openErr)
			return
		}

		defer cfgReader.Close()

		if readErr := viper.ReadConfig(cfgReader); readErr != nil {
			fmt.Printf("embedded config file not readable: %v\n", readErr)
			return
		}
	}

	viper.WatchConfig()
}

const rootLong = `
Caramba is a fully featured machine learning research platform.
`
