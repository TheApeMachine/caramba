package environment

import (
	"archive/tar"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"time"

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

	var wg sync.WaitGroup

	if stdin != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			io.Copy(resp.Conn, stdin)
			resp.CloseWrite()
		}()
	}

	if stdout != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			io.Copy(stdout, resp.Reader)
		}()
	}

	// Start a goroutine to wait for all IO operations to complete
	go func() {
		wg.Wait()
		resp.Close()
	}()

	return nil
}

// demultiplexDockerStream takes a Docker multiplexed stream and writes stdout/stderr to the appropriate writers
func demultiplexDockerStream(reader io.Reader, stdout, stderr io.Writer) error {
	var (
		header = make([]byte, 8)
		err    error
	)

	for {
		// Read header
		_, err = io.ReadFull(reader, header)
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}

		// Get size of the coming message
		size := int64(header[4])<<24 | int64(header[5])<<16 | int64(header[6])<<8 | int64(header[7])

		// Choose writer based on stream type (header[0])
		var w io.Writer
		switch header[0] {
		case 1:
			w = stdout
		case 2:
			w = stderr
		default:
			continue
		}

		// Copy the message to the appropriate writer
		_, err = io.CopyN(w, reader, size)
		if err != nil {
			return err
		}
	}
}

func (runtime *dockerRuntime) ExecuteCommand(ctx context.Context, command string, stdout, stderr io.Writer) error {
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

	// Attach to the exec instance to get the output
	resp, err := runtime.client.ContainerExecAttach(ctx, exec.ID, container.ExecStartOptions{})
	if err != nil {
		return errnie.Error(err)
	}
	defer resp.Close()

	// Start the command
	if err := runtime.client.ContainerExecStart(ctx, exec.ID, container.ExecStartOptions{}); err != nil {
		return errnie.Error(err)
	}

	// Create buffers to capture output
	var stdoutBuf, stderrBuf bytes.Buffer
	mw := io.MultiWriter(stdout, &stdoutBuf)
	mwErr := io.MultiWriter(stderr, &stderrBuf)

	// Copy output using the demultiplexer since we're not in TTY mode
	errCh := make(chan error, 1)
	go func() {
		errCh <- demultiplexDockerStream(resp.Reader, mw, mwErr)
	}()

	// Wait for the command to complete
	for {
		inspectResp, err := runtime.client.ContainerExecInspect(ctx, exec.ID)
		if err != nil {
			return errnie.Error(err)
		}
		if !inspectResp.Running {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	// Wait for output copying to complete
	copyErr := <-errCh
	if copyErr != nil && copyErr != io.EOF {
		return errnie.Error(copyErr)
	}

	// Check if this was an EOF during input read
	if stderrStr := stderrBuf.String(); stderrStr != "" &&
		(stderrStr == "EOFError: EOF when reading a line\n" ||
			stderrStr == "EOFError: EOF when reading a line") {
		// Just return without error - this is expected for interactive programs
		return nil
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
