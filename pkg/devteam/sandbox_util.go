package devteam

import (
	"archive/tar"
	"bytes"
	"net/url"
	"path"
	"strings"

	"github.com/docker/docker/api/types/container"
)

func (sandbox *Sandbox) copyFile(targetPath, content string, mode int64) error {
	var buf bytes.Buffer
	tarWriter := tar.NewWriter(&buf)

	header := &tar.Header{
		Name: path.Base(targetPath),
		Mode: mode,
		Size: int64(len(content)),
	}

	if err := tarWriter.WriteHeader(header); err != nil {
		return err
	}

	if _, err := tarWriter.Write([]byte(content)); err != nil {
		return err
	}

	if err := tarWriter.Close(); err != nil {
		return err
	}

	return sandbox.docker.CopyToContainer(
		sandbox.ctx,
		sandbox.containerID,
		path.Dir(targetPath),
		&buf,
		container.CopyToContainerOptions{},
	)
}

/*
Redact removes sensitive values from text before it can enter logs, errors, or
kanban card descriptions.
*/
func (sandbox *Sandbox) Redact(text string) string {
	return redactSecrets(text, sandbox.cfg.GitHubToken)
}

func combinedOutput(stdout, stderr string) string {
	if stderr == "" {
		return stdout
	}

	if stdout == "" {
		return stderr
	}

	return stdout + "\n" + stderr
}

func redactSecrets(text string, secrets ...string) string {
	redacted := text

	for _, secret := range secrets {
		if secret == "" {
			continue
		}

		redacted = strings.ReplaceAll(redacted, secret, "[REDACTED]")
		redacted = strings.ReplaceAll(redacted, url.QueryEscape(secret), "[REDACTED]")
		redacted = strings.ReplaceAll(redacted, url.PathEscape(secret), "[REDACTED]")
	}

	return redacted
}
