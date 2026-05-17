package devteam

import (
	"archive/tar"
	"bytes"
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/client"
	"github.com/docker/docker/pkg/stdcopy"
)

/*
Sandbox manages a single ephemeral Docker container that hosts one feature
branch. It provides an exec interface so agents can run arbitrary shell
commands inside the container.

Lifecycle: New → Start (pulls image, creates container, clones repo, branches)
→ Exec (developer/reviewer work) → Destroy (stops + removes container).
*/
type Sandbox struct {
	ctx         context.Context
	docker      *client.Client
	containerID string
	cfg         SandboxConfig
}

/*
NewSandbox allocates a Sandbox but does not yet start the container.
*/
func NewSandbox(ctx context.Context, cfg SandboxConfig) (*Sandbox, error) {
	docker, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())

	if err != nil {
		return nil, fmt.Errorf("sandbox: docker client: %w", err)
	}

	return &Sandbox{
		ctx:    ctx,
		docker: docker,
		cfg:    cfg,
	}, nil
}

/*
Start pulls the image if needed, creates the container, injects Git credentials,
clones the repository, and checks out a fresh feature branch.
*/
func (sandbox *Sandbox) Start() error {
	if err := sandbox.ensureImage(); err != nil {
		return err
	}

	if err := sandbox.createContainer(); err != nil {
		return err
	}

	if err := sandbox.docker.ContainerStart(sandbox.ctx, sandbox.containerID, container.StartOptions{}); err != nil {
		return fmt.Errorf("sandbox: start container: %w", err)
	}

	if err := sandbox.connectNetwork(); err != nil {
		return err
	}

	defer sandbox.disconnectNetwork()

	if err := sandbox.installGitCredentials(); err != nil {
		return err
	}

	setup := []string{
		"git --version",
		"go version",
		fmt.Sprintf(
			`git clone --depth=1 "https://github.com/%s/%s.git" /workspace`,
			sandbox.cfg.GitHubOwner, sandbox.cfg.GitHubRepo,
		),
		fmt.Sprintf(`git -C /workspace checkout -b %s`, sandbox.cfg.FeatureBranch),
		`git -C /workspace config user.email "devteam@caramba.ai"`,
		`git -C /workspace config user.name "Caramba Dev Team"`,
	}

	for _, cmd := range setup {
		if _, err := sandbox.Exec(cmd); err != nil {
			return fmt.Errorf("sandbox: setup %q: %w", truncate(sandbox.Redact(cmd), 40), err)
		}
	}

	return nil
}

/*
Exec runs a shell command inside the container and returns combined stdout+stderr.
*/
func (sandbox *Sandbox) Exec(command string) (string, error) {
	timeout := sandbox.execTimeout()
	execCtx, cancel := context.WithTimeout(sandbox.ctx, timeout+5*time.Second)
	defer cancel()

	execID, err := sandbox.docker.ContainerExecCreate(execCtx, sandbox.containerID, container.ExecOptions{
		Cmd: []string{
			"timeout",
			"--kill-after=5s",
			fmt.Sprintf("%ds", int(timeout.Seconds())),
			"/bin/sh",
			"-c",
			command,
		},
		AttachStdout: true,
		AttachStderr: true,
		WorkingDir:   "/workspace",
	})

	if err != nil {
		return "", fmt.Errorf("sandbox: exec create: %w", err)
	}

	resp, err := sandbox.docker.ContainerExecAttach(execCtx, execID.ID, container.ExecStartOptions{})

	if err != nil {
		return "", fmt.Errorf("sandbox: exec attach: %w", err)
	}

	defer resp.Close()

	var stdout bytes.Buffer
	var stderr bytes.Buffer

	if _, err := stdcopy.StdCopy(&stdout, &stderr, resp.Reader); err != nil && err != io.EOF {
		return "", fmt.Errorf("sandbox: exec read: %w", err)
	}

	inspect, err := sandbox.docker.ContainerExecInspect(execCtx, execID.ID)
	output := sandbox.Redact(combinedOutput(stdout.String(), stderr.String()))

	if err != nil {
		return output, fmt.Errorf("sandbox: exec inspect: %w", err)
	}

	if inspect.ExitCode != 0 {
		if inspect.ExitCode == 124 {
			return output, fmt.Errorf(
				"sandbox: exec timed out after %s: %s",
				timeout, truncate(output, 400),
			)
		}

		return output, fmt.Errorf(
			"sandbox: command exited %d: %s",
			inspect.ExitCode, truncate(output, 400),
		)
	}

	return output, nil
}

/*
ReadFile returns the content of a file inside the container's /workspace.
*/
func (sandbox *Sandbox) ReadFile(path string) (string, error) {
	reader, _, err := sandbox.docker.CopyFromContainer(
		sandbox.ctx, sandbox.containerID, "/workspace/"+strings.TrimPrefix(path, "/"),
	)

	if err != nil {
		return "", fmt.Errorf("sandbox: read file %q: %w", path, err)
	}

	defer reader.Close()

	tr := tar.NewReader(reader)

	if _, err := tr.Next(); err != nil {
		return "", fmt.Errorf("sandbox: tar next: %w", err)
	}

	var buf bytes.Buffer

	if _, err := io.Copy(&buf, tr); err != nil {
		return "", fmt.Errorf("sandbox: tar read: %w", err)
	}

	return buf.String(), nil
}

/*
WriteFile creates or overwrites a file inside the container's /workspace.
*/
func (sandbox *Sandbox) WriteFile(path, content string) error {
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)

	hdr := &tar.Header{
		Name: strings.TrimPrefix(path, "/"),
		Mode: 0644,
		Size: int64(len(content)),
	}

	if err := tw.WriteHeader(hdr); err != nil {
		return err
	}

	if _, err := tw.Write([]byte(content)); err != nil {
		return err
	}

	if err := tw.Close(); err != nil {
		return err
	}

	return sandbox.docker.CopyToContainer(
		sandbox.ctx, sandbox.containerID, "/workspace", &buf, container.CopyToContainerOptions{},
	)
}

/*
CommitAndPush stages all changes, commits with the given message, rebases onto
the remote feature branch, and pushes with a lease.
*/
func (sandbox *Sandbox) CommitAndPush(message string) error {
	if err := sandbox.connectNetwork(); err != nil {
		return err
	}

	defer sandbox.disconnectNetwork()

	commands := []string{
		`git -C /workspace add -A`,
		fmt.Sprintf(`git -C /workspace commit -m %q`, message),
	}

	for _, cmd := range commands {
		if _, err := sandbox.Exec(cmd); err != nil {
			return fmt.Errorf("sandbox: git %q: %w", truncate(sandbox.Redact(cmd), 80), err)
		}
	}

	if err := sandbox.rebaseRemoteBranch(); err != nil {
		return err
	}

	cmd := fmt.Sprintf(`git -C /workspace push --force-with-lease origin %s`, sandbox.cfg.FeatureBranch)

	if _, err := sandbox.Exec(cmd); err != nil {
		return fmt.Errorf("sandbox: git %q: %w", truncate(sandbox.Redact(cmd), 80), err)
	}

	return nil
}

/*
Destroy stops and removes the container, releasing all resources.
*/
func (sandbox *Sandbox) Destroy() error {
	if sandbox.containerID == "" {
		return nil
	}

	_ = sandbox.docker.ContainerStop(sandbox.ctx, sandbox.containerID, container.StopOptions{})

	return sandbox.docker.ContainerRemove(sandbox.ctx, sandbox.containerID, container.RemoveOptions{
		Force: true,
	})
}

func (sandbox *Sandbox) ensureImage() error {
	reader, err := sandbox.docker.ImagePull(sandbox.ctx, sandbox.cfg.Image, image.PullOptions{})

	if err != nil {
		return fmt.Errorf("sandbox: pull image %q: %w", sandbox.cfg.Image, err)
	}

	defer reader.Close()
	_, err = io.Copy(io.Discard, reader)

	return err
}

func (sandbox *Sandbox) createContainer() error {
	resp, err := sandbox.docker.ContainerCreate(
		sandbox.ctx,
		&container.Config{
			Image: sandbox.cfg.Image,
			Cmd:   []string{"/bin/sh", "-c", "tail -f /dev/null"},
			Tty:   false,
		},
		sandbox.hostConfig(),
		nil,
		nil,
		"",
	)

	if err != nil {
		return fmt.Errorf("sandbox: create container: %w", err)
	}

	sandbox.containerID = resp.ID

	return nil
}

/*
PatchFile applies a unified diff (produced by diff -u or git diff) to the
given path inside /workspace. The diff is written to a temp file and applied
with patch -p1 so relative paths strip the a/ b/ prefixes automatically.
*/
func (sandbox *Sandbox) PatchFile(unifiedDiff string) error {
	escaped := strings.ReplaceAll(unifiedDiff, "'", `'\''`)
	cmd := fmt.Sprintf(
		`printf '%%s' '%s' | patch -p1 --no-backup-if-mismatch -d /workspace`,
		escaped,
	)

	_, err := sandbox.Exec(cmd)

	return err
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}

	return s[:n] + "…"
}
