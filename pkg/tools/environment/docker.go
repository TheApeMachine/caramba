package environment

import (
	"archive/tar"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"

	"slices"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/client"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/fs"
	"github.com/theapemachine/caramba/pkg/workflow"
)

type dockerRuntime struct {
	client      *client.Client
	containerID string
}

func newDockerRuntime() (Runtime, error) {
	cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return nil, fmt.Errorf("failed to create Docker client: %w", err)
	}

	return &dockerRuntime{
		client: cli,
	}, nil
}

func (runtime *dockerRuntime) CreateContainer(ctx context.Context) (err error) {
	containerName := "caramba-env"

	// Check if container already exists
	containers, err := runtime.client.ContainerList(ctx, container.ListOptions{All: true})

	if err != nil {
		return errnie.Error(err)
	}

	// Look for our container
	for _, container := range containers {
		if slices.Contains(container.Names, "/"+containerName) {
			runtime.containerID = container.ID
			return nil
		}
	}

	// If we get here, container doesn't exist, so create it
	var (
		artifact = datura.New()
		payload  []byte
	)

	// Get the Dockerfile content from the artifact
	if _, err := io.Copy(artifact, workflow.NewPipeline(datura.New(
		datura.WithRole(datura.ArtifactRoleOpenFile),
		datura.WithMeta("path", "manifests/Dockerfile"),
	), fs.NewStore())); err != nil {
		return errnie.Error(err)
	}

	if payload, err = artifact.DecryptPayload(); err != nil {
		return errnie.Error(err)
	}

	// Build the image using the Dockerfile
	if err := runtime.BuildImage(ctx, payload, containerName); err != nil {
		return errnie.Error(err)
	}

	// Create container using our built image
	resp, err := runtime.client.ContainerCreate(ctx,
		&container.Config{
			Image: containerName,
			Cmd:   []string{"/bin/bash"},
			Tty:   true,
		},
		nil, nil, nil, containerName,
	)

	if err != nil {
		return errnie.Error(err)
	}

	runtime.containerID = resp.ID
	return nil
}

func (runtime *dockerRuntime) StartContainer(ctx context.Context) (err error) {
	if err = runtime.client.ContainerStart(ctx, runtime.containerID, container.StartOptions{}); err != nil {
		return errnie.Error(err)
	}

	return nil
}

func (runtime *dockerRuntime) StopContainer(ctx context.Context) (err error) {
	if err = runtime.client.ContainerStop(ctx, runtime.containerID, container.StopOptions{}); err != nil {
		return errnie.Error(err)
	}

	return nil
}

func (runtime *dockerRuntime) AttachIO(stdin io.Reader, stdout, stderr io.Writer) error {
	resp, err := runtime.client.ContainerAttach(
		context.Background(),
		runtime.containerID,
		container.AttachOptions{
			Stream: true,
			Stdin:  stdin != nil,
			Stdout: stdout != nil,
			Stderr: stderr != nil,
			Logs:   true,
		},
	)

	if err != nil {
		return errnie.Error(err)
	}

	go func() {
		if stdin != nil {
			if _, err := io.Copy(resp.Conn, stdin); err != nil {
				errnie.Error(err)
			}
		}
	}()

	go func() {
		if stdout != nil {
			if _, err := io.Copy(stdout, resp.Reader); err != nil {
				errnie.Error(err)
			}
		}
	}()

	return nil
}

func (runtime *dockerRuntime) ExecuteCommand(ctx context.Context, command string) error {
	exec, err := runtime.client.ContainerExecCreate(
		ctx,
		runtime.containerID,
		container.ExecOptions{
			Cmd:          []string{"/bin/sh", "-c", command},
			AttachStdout: true,
			AttachStderr: true,
		},
	)

	if err != nil {
		return errnie.Error(err)
	}

	if err := runtime.client.ContainerExecStart(ctx, exec.ID, container.ExecStartOptions{}); err != nil {
		return errnie.Error(err)
	}

	return nil
}

func (runtime *dockerRuntime) PullImage(ctx context.Context, ref string) error {
	errnie.Debug(fmt.Sprintf("Pulling image: %s", ref))

	reader, err := runtime.client.ImagePull(ctx, ref, image.PullOptions{})

	if err != nil {
		return errnie.Error(err)
	}

	defer reader.Close()

	// Read the output to complete the pull
	_, err = io.Copy(io.Discard, reader)
	return errnie.Error(err)
}

func (runtime *dockerRuntime) BuildImage(
	ctx context.Context, dockerfile []byte, imageName string,
) error {
	errnie.Debug(fmt.Sprintf("Building image: %s", imageName))

	// Create a buffer to store our tar archive
	var buf bytes.Buffer

	// Create a new tar writer
	tw := tar.NewWriter(&buf)

	// Create a tar header for the Dockerfile
	header := &tar.Header{
		Name: "Dockerfile",
		Mode: 0600,
		Size: int64(len(dockerfile)),
	}

	// Write the header
	if err := tw.WriteHeader(header); err != nil {
		return errnie.Error(err)
	}

	// Write the Dockerfile content
	if _, err := tw.Write(dockerfile); err != nil {
		return errnie.Error(err)
	}

	// Close the tar writer
	if err := tw.Close(); err != nil {
		return errnie.Error(err)
	}

	opts := types.ImageBuildOptions{
		Dockerfile: "Dockerfile",
		Tags:       []string{imageName},
		Remove:     true,
		BuildArgs: map[string]*string{
			"TARGETARCH": nil, // This will use the default architecture
		},
	}

	resp, err := runtime.client.ImageBuild(ctx, &buf, opts)
	if err != nil {
		return errnie.Error(err)
	}

	defer resp.Body.Close()

	// Read the output to complete the build
	return runtime.processAndPrintBuildOutput(resp.Body)
}

func (runtime *dockerRuntime) processAndPrintBuildOutput(reader io.Reader) error {
	decoder := json.NewDecoder(reader)
	for {
		var message struct {
			Stream string `json:"stream"`
			Error  string `json:"error"`
		}

		if err := decoder.Decode(&message); err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}

		if message.Error != "" {
			return errnie.Error(fmt.Errorf("build error: %s", message.Error))
		}

		if message.Stream != "" {
			fmt.Print(message.Stream)
		}
	}
}
