package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"text/template"
	"time"

	"github.com/spf13/cobra"
)

var researchCmd = &cobra.Command{
	Use:   "research <name>",
	Short: "Scaffold a new research project.",
	Args:  cobra.ExactArgs(1),
	RunE:  runResearch,
}

func init() {
	rootCmd.AddCommand(researchCmd)
}

func runResearch(cmd *cobra.Command, args []string) error {
	name := args[0]
	root := filepath.Join("research", "project", name)

	dirs := []string{
		filepath.Join(root, "manifest", "architecture"),
		filepath.Join(root, "manifest", "operation"),
		filepath.Join(root, "paper"),
	}

	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("research: cannot create %s: %w", dir, err)
		}
	}

	data := templateData{
		Name:        name,
		Description: name,
		Created:     time.Now().Format(time.RFC3339),
		Updated:     time.Now().Format(time.RFC3339),
	}

	return renderTemplate("asset/template/manifest/project.yml", filepath.Join(root, "manifest", "master.yml"), data)
}

func renderTemplate(assetPath, dest string, data templateData) error {
	raw, err := embedded.ReadFile(assetPath)

	if err != nil {
		return fmt.Errorf("research: cannot read template %s: %w", assetPath, err)
	}

	tmpl, err := template.New("").Parse(string(raw))

	if err != nil {
		return fmt.Errorf("research: cannot parse template %s: %w", assetPath, err)
	}

	f, err := os.Create(dest)

	if err != nil {
		return fmt.Errorf("research: cannot create %s: %w", dest, err)
	}

	defer f.Close()

	return tmpl.Execute(f, data)
}

type templateData struct {
	Name        string
	Description string
	Created     string
	Updated     string
	Topology    string
	Nodes       string
	Edges       string
	Source      string
	Target      string
	Type        string
	Inputs      string
	Outputs     string
}
