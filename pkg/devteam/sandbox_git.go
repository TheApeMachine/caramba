package devteam

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/docker/docker/api/types/network"
)

func (sandbox *Sandbox) connectNetwork() error {
	if err := sandbox.docker.NetworkConnect(
		sandbox.ctx,
		sandboxGitNetwork,
		sandbox.containerID,
		&network.EndpointSettings{},
	); err != nil {
		return fmt.Errorf("sandbox: connect network: %w", err)
	}

	return nil
}

func (sandbox *Sandbox) disconnectNetwork() {
	_ = sandbox.docker.NetworkDisconnect(
		sandbox.ctx,
		sandboxGitNetwork,
		sandbox.containerID,
		false,
	)
}

func (sandbox *Sandbox) installGitCredentials() error {
	credentials := fmt.Sprintf(
		"https://%s@github.com\n",
		url.UserPassword("x-access-token", sandbox.cfg.GitHubToken).String(),
	)

	if err := sandbox.copyFile("/root/.git-credentials", credentials, 0600); err != nil {
		return fmt.Errorf("sandbox: git credentials: %w", err)
	}

	if _, err := sandbox.Exec(`git config --global credential.helper store`); err != nil {
		return fmt.Errorf("sandbox: git credential helper: %w", err)
	}

	return nil
}

func (sandbox *Sandbox) rebaseRemoteBranch() error {
	out, err := sandbox.Exec(fmt.Sprintf(
		`git -C /workspace ls-remote --heads origin %s`,
		sandbox.cfg.FeatureBranch,
	))

	if err != nil {
		return fmt.Errorf("sandbox: inspect remote branch: %w", err)
	}

	if strings.TrimSpace(out) == "" {
		return nil
	}

	commands := []string{
		fmt.Sprintf(
			`git -C /workspace fetch origin %s:refs/remotes/origin/%s`,
			sandbox.cfg.FeatureBranch,
			sandbox.cfg.FeatureBranch,
		),
		fmt.Sprintf(
			`git -C /workspace rebase refs/remotes/origin/%s`,
			sandbox.cfg.FeatureBranch,
		),
	}

	for _, cmd := range commands {
		if _, err := sandbox.Exec(cmd); err != nil {
			return fmt.Errorf("sandbox: rebase %q: %w", truncate(sandbox.Redact(cmd), 80), err)
		}
	}

	return nil
}
