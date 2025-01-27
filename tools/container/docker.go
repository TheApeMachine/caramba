package container

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/docker/docker/api/types"
	dc "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/api/types/network"
	"github.com/docker/docker/client"
	"github.com/docker/docker/pkg/archive"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/theapemachine/errnie"
)

type Docker struct {
	ID     string
	cli    *client.Client
	vendor string
	name   string
	tag    string
	err    error
}

func NewDocker(
	ctx context.Context, vendor, name string, tag string,
) *Docker {
	var (
		cli *client.Client
		err error
	)

	if cli, err = client.NewClientWithOpts(
		client.FromEnv, client.WithAPIVersionNegotiation(),
	); errnie.Error(err) != nil {
		return nil
	}

	return &Docker{"", cli, vendor, name, tag, nil}
}

func (docker *Docker) Start() *Docker {
	var (
		err error
		out io.ReadCloser
	)

	if err = docker.cli.ContainerStart(
		context.Background(), docker.ID, dc.StartOptions{},
	); errnie.Error(err) != nil {
		return docker
	}

	statusCh, errCh := docker.cli.ContainerWait(
		context.Background(),
		docker.ID,
		dc.WaitConditionNotRunning,
	)

	select {
	case err = <-errCh:
		if err != nil {
			return docker
		}
	case <-statusCh:
	}

	if out, err = docker.cli.ContainerLogs(
		context.Background(), docker.ID, dc.LogsOptions{
			ShowStdout: true,
		},
	); err != nil {
		return docker
	}

	io.Copy(os.Stdout, out)
	return docker
}

func (docker *Docker) Create(entrypoint, cmd *[]string) *Docker {
	var (
		reader dc.CreateResponse
		err    error
	)

	config := &dc.Config{Tty: true}

	if entrypoint != nil {
		config.Entrypoint = *entrypoint
	}

	if cmd != nil {
		config.Cmd = *cmd
	}

	if reader, err = docker.cli.ContainerCreate(
		context.Background(),
		config,
		&dc.HostConfig{
			Binds: []string{fmt.Sprintf(
				"~/tmp/wrkspc/%s/%s", docker.vendor, docker.name,
			)},
		},
		&network.NetworkingConfig{},
		&specs.Platform{},
		"",
	); err != nil {
		return docker
	}

	docker.ID = reader.ID

	return docker
}

func (docker *Docker) Pull() *Docker {
	var (
		reader io.ReadCloser
		err    error
	)

	tags := []string{fmt.Sprintf(
		"%s/%s", docker.vendor, docker.name,
	)}

	if docker.tag != "" {
		tags = append(tags, ":"+docker.tag)
	}

	if reader, err = docker.cli.ImagePull(
		context.Background(),
		strings.Join(tags, ""),
		image.PullOptions{},
	); errnie.Error(err) != nil {
		return docker
	}

	io.Copy(os.Stdout, reader)
	return docker
}

func (docker *Docker) Build() *Docker {
	var (
		tar io.ReadCloser
		res types.ImageBuildResponse
	)

	// Set path to current workdir.
	wd, err := os.Getwd()
	if err != nil {
		return docker
	}

	if tar, docker.err = archive.TarWithOptions(
		wd, &archive.TarOptions{},
	); errnie.Error(docker.err) != nil {
		return docker
	}

	tags := []string{fmt.Sprintf(
		"%s/%s", docker.vendor, docker.name,
	)}

	if docker.tag != "" {
		tags = append(tags, ":"+docker.tag)
	}

	if res, docker.err = docker.cli.ImageBuild(
		context.Background(), tar, types.ImageBuildOptions{
			Dockerfile: "Dockerfile",
			Tags:       tags,
			Remove:     true,
			PullParent: true,
			Platform:   "linux/amd64",
		},
	); errnie.Error(docker.err) != nil {
		return docker
	}

	scanner := bufio.NewScanner(res.Body)

	for scanner.Scan() {
		fmt.Println(scanner.Text())
	}

	return docker
}
