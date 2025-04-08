package editor

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"

	"github.com/theapemachine/caramba/pkg/datura"
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

func (client *Client) encode(artifact datura.Artifact, v any) (err error) {
	errnie.Debug("editor.Client.encode")

	payload := bytes.NewBuffer([]byte{})

	if err = json.NewEncoder(payload).Encode(v); err != nil {
		return errnie.Error(err)
	}

	artifact = datura.WithEncryptedPayload(payload.Bytes())(artifact)
	return nil
}

func (client *Client) ReadFile(artifact datura.Artifact) error {
	errnie.Debug("editor.Client.ReadFile")

	filePath := datura.GetMetaValue[string](artifact, "file")
	content, err := os.ReadFile(filePath)
	if err != nil {
		return errnie.Error(err)
	}

	return client.encode(artifact, map[string]any{
		"content": string(content),
	})
}

func (client *Client) WriteFile(artifact datura.Artifact) error {
	errnie.Debug("editor.Client.WriteFile")

	filePath := datura.GetMetaValue[string](artifact, "file")
	content := datura.GetMetaValue[string](artifact, "content")

	err := os.WriteFile(filePath, []byte(content), 0644)
	if err != nil {
		return errnie.Error(err)
	}

	return client.encode(artifact, map[string]any{
		"success": true,
	})
}

func (client *Client) DeleteFile(artifact datura.Artifact) error {
	errnie.Debug("editor.Client.DeleteFile")

	filePath := datura.GetMetaValue[string](artifact, "file")

	err := os.Remove(filePath)
	if err != nil {
		return errnie.Error(err)
	}

	return client.encode(artifact, map[string]any{
		"success": true,
	})
}

func (client *Client) ReplaceLines(artifact datura.Artifact) error {
	errnie.Debug("editor.Client.ReplaceLines")

	filePath := datura.GetMetaValue[string](artifact, "file")
	content := datura.GetMetaValue[string](artifact, "content")
	startLine := datura.GetMetaValue[int](artifact, "start_line")
	endLine := datura.GetMetaValue[int](artifact, "end_line")

	// Read the entire file
	fileContent, err := os.ReadFile(filePath)
	if err != nil {
		return errnie.Error(err)
	}

	lines := strings.Split(string(fileContent), "\n")

	// Validate line numbers
	if startLine < 1 || endLine > len(lines) || startLine > endLine {
		return errnie.Error(err)
	}

	// Replace the specified lines
	newLines := strings.Split(content, "\n")
	lines = append(lines[:startLine-1], append(newLines, lines[endLine:]...)...)

	// Write back to file
	err = os.WriteFile(filePath, []byte(strings.Join(lines, "\n")), 0644)
	if err != nil {
		return errnie.Error(err)
	}

	return client.encode(artifact, map[string]any{
		"success": true,
	})
}

func (client *Client) InsertLines(artifact datura.Artifact) error {
	errnie.Debug("editor.Client.InsertLines")

	filePath := datura.GetMetaValue[string](artifact, "file")
	content := datura.GetMetaValue[string](artifact, "content")
	lineNumber := datura.GetMetaValue[int](artifact, "line_number")

	// Read the entire file
	fileContent, err := os.ReadFile(filePath)
	if err != nil {
		return errnie.Error(err)
	}

	lines := strings.Split(string(fileContent), "\n")

	// Validate line number
	if lineNumber < 1 || lineNumber > len(lines)+1 {
		return errnie.Error(err)
	}

	// Insert the new lines
	newLines := strings.Split(content, "\n")
	lines = append(lines[:lineNumber-1], append(newLines, lines[lineNumber-1:]...)...)

	// Write back to file
	err = os.WriteFile(filePath, []byte(strings.Join(lines, "\n")), 0644)
	if err != nil {
		return errnie.Error(err)
	}

	return client.encode(artifact, map[string]any{
		"success": true,
	})
}

func (client *Client) DeleteLines(artifact datura.Artifact) error {
	errnie.Debug("editor.Client.DeleteLines")

	filePath := datura.GetMetaValue[string](artifact, "file")
	startLine := datura.GetMetaValue[int](artifact, "start_line")
	endLine := datura.GetMetaValue[int](artifact, "end_line")

	// Read the entire file
	fileContent, err := os.ReadFile(filePath)
	if err != nil {
		return errnie.Error(err)
	}

	lines := strings.Split(string(fileContent), "\n")

	// Validate line numbers
	if startLine < 1 || endLine > len(lines) || startLine > endLine {
		return errnie.Error(err)
	}

	// Remove the specified lines
	lines = append(lines[:startLine-1], lines[endLine:]...)

	// Write back to file
	err = os.WriteFile(filePath, []byte(strings.Join(lines, "\n")), 0644)
	if err != nil {
		return errnie.Error(err)
	}

	return client.encode(artifact, map[string]any{
		"success": true,
	})
}

func (client *Client) ReadLines(artifact datura.Artifact) error {
	errnie.Debug("editor.Client.ReadLines")

	filePath := datura.GetMetaValue[string](artifact, "file")
	startLine := datura.GetMetaValue[int](artifact, "start_line")
	endLine := datura.GetMetaValue[int](artifact, "end_line")

	// Read the entire file
	fileContent, err := os.ReadFile(filePath)
	if err != nil {
		return errnie.Error(err)
	}

	lines := strings.Split(string(fileContent), "\n")

	// Validate line numbers
	if startLine < 1 || endLine > len(lines) || startLine > endLine {
		return errnie.Error(err)
	}

	// Extract the specified lines
	selectedLines := lines[startLine-1 : endLine]

	return client.encode(artifact, map[string]any{
		"content": strings.Join(selectedLines, "\n"),
	})
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
