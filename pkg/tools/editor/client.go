package editor

import (
	"bytes"
	"os"
	"strings"

	"github.com/theapemachine/caramba/pkg/errnie"
)

type Client struct {
	buffer *bytes.Buffer
}

func NewClient() *Client {
	errnie.Debug("editor.NewClient")

	return &Client{
		buffer: bytes.NewBuffer([]byte{}),
	}
}

func (client *Client) ReadFile(artifact map[string]any) error {
	errnie.Debug("editor.Client.ReadFile")

	filePath := artifact["file"].(string)
	content, err := os.ReadFile(filePath)
	if err != nil {
		return errnie.New(errnie.WithError(err))
	}

	artifact["content"] = string(content)

	return nil
}

func (client *Client) WriteFile(artifact map[string]any) error {
	errnie.Debug("editor.Client.WriteFile")

	filePath := artifact["file"].(string)
	content := artifact["content"].(string)

	err := os.WriteFile(filePath, []byte(content), 0644)
	if err != nil {
		return errnie.New(errnie.WithError(err))
	}

	artifact["success"] = true

	return nil
}

func (client *Client) DeleteFile(artifact map[string]any) error {
	errnie.Debug("editor.Client.DeleteFile")

	filePath := artifact["file"].(string)

	err := os.Remove(filePath)
	if err != nil {
		return errnie.New(errnie.WithError(err))
	}

	artifact["success"] = true

	return nil
}

func (client *Client) ReplaceLines(artifact map[string]any) error {
	errnie.Debug("editor.Client.ReplaceLines")

	filePath := artifact["file"].(string)
	content := artifact["content"].(string)
	startLine := artifact["start_line"].(int)
	endLine := artifact["end_line"].(int)

	// Read the entire file
	fileContent, err := os.ReadFile(filePath)
	if err != nil {
		return errnie.New(errnie.WithError(err))
	}

	lines := strings.Split(string(fileContent), "\n")

	// Validate line numbers
	if startLine < 1 || endLine > len(lines) || startLine > endLine {
		return errnie.New(errnie.WithError(err))
	}

	// Replace the specified lines
	newLines := strings.Split(content, "\n")
	lines = append(lines[:startLine-1], append(newLines, lines[endLine:]...)...)

	// Write back to file
	err = os.WriteFile(filePath, []byte(strings.Join(lines, "\n")), 0644)
	if err != nil {
		return errnie.New(errnie.WithError(err))
	}

	artifact["success"] = true

	return nil
}

func (client *Client) InsertLines(artifact map[string]any) error {
	errnie.Debug("editor.Client.InsertLines")

	filePath := artifact["file"].(string)
	content := artifact["content"].(string)
	lineNumber := artifact["line_number"].(int)

	// Read the entire file
	fileContent, err := os.ReadFile(filePath)
	if err != nil {
		return errnie.New(errnie.WithError(err))
	}

	lines := strings.Split(string(fileContent), "\n")

	// Validate line number
	if lineNumber < 1 || lineNumber > len(lines)+1 {
		return errnie.New(errnie.WithError(err))
	}

	// Insert the new lines
	newLines := strings.Split(content, "\n")
	lines = append(lines[:lineNumber-1], append(newLines, lines[lineNumber-1:]...)...)

	// Write back to file
	err = os.WriteFile(filePath, []byte(strings.Join(lines, "\n")), 0644)
	if err != nil {
		return errnie.New(errnie.WithError(err))
	}

	artifact["success"] = true

	return nil
}

func (client *Client) DeleteLines(artifact map[string]any) error {
	errnie.Debug("editor.Client.DeleteLines")

	filePath := artifact["file"].(string)
	startLine := artifact["start_line"].(int)
	endLine := artifact["end_line"].(int)

	// Read the entire file
	fileContent, err := os.ReadFile(filePath)
	if err != nil {
		return errnie.New(errnie.WithError(err))
	}

	lines := strings.Split(string(fileContent), "\n")

	// Validate line numbers
	if startLine < 1 || endLine > len(lines) || startLine > endLine {
		return errnie.New(errnie.WithError(err))
	}

	// Remove the specified lines
	lines = append(lines[:startLine-1], lines[endLine:]...)

	// Write back to file
	err = os.WriteFile(filePath, []byte(strings.Join(lines, "\n")), 0644)
	if err != nil {
		return errnie.New(errnie.WithError(err))
	}

	artifact["success"] = true

	return nil
}

func (client *Client) ReadLines(artifact map[string]any) error {
	errnie.Debug("editor.Client.ReadLines")

	filePath := artifact["file"].(string)
	startLine := artifact["start_line"].(int)
	endLine := artifact["end_line"].(int)

	// Read the entire file
	fileContent, err := os.ReadFile(filePath)
	if err != nil {
		return errnie.New(errnie.WithError(err))
	}

	lines := strings.Split(string(fileContent), "\n")

	// Validate line numbers
	if startLine < 1 || endLine > len(lines) || startLine > endLine {
		return errnie.New(errnie.WithError(err))
	}

	// Extract the specified lines
	selectedLines := lines[startLine-1 : endLine]

	artifact["content"] = strings.Join(selectedLines, "\n")

	return nil
}

func (client *Client) Read(p []byte) (n int, err error) {
	errnie.Debug("editor.Client.Read")
	return client.buffer.Read(p)
}

func (client *Client) Write(p []byte) (n int, err error) {
	errnie.Debug("editor.Client.Write")
	return client.buffer.Write(p)
}

func (client *Client) Close() error {
	errnie.Debug("editor.Client.Close")
	client.buffer.Reset()
	return nil
}
