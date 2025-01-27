package container

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"

	"github.com/charmbracelet/log"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/client"
	"github.com/docker/docker/pkg/archive"
)

/*
Builder is a wrapper around the Docker client that provides methods for building
and running containers. It encapsulates the complexity of Docker operations,
allowing for easier management of containerized environments.
*/
type Builder struct {
	client *client.Client
}

/*
NewBuilder creates a new Builder instance.
It initializes a Docker client using the host's Docker environment settings.
*/
func NewBuilder() *Builder {
	cli, err := client.NewClientWithOpts(client.FromEnv)
	if err != nil {
		return nil
	}
	return &Builder{client: cli}
}

// DefaultImageName is the standard name used for the container image
const DefaultImageName = "caramba-terminal:latest"

/*
BuildImage constructs a Docker image from a Dockerfile in the specified directory.
This method abstracts the image building process, handling the creation of a tar archive
and the configuration of build options.
*/
func (b *Builder) BuildImage(ctx context.Context, dockerfilePath, imageName string) error {
	log.Info("Building image", "dockerfilePath", dockerfilePath, "imageName", imageName)

	// Get the directory containing the Dockerfile
	dockerfileDir := filepath.Dir(dockerfilePath)

	// Create tar of the directory containing Dockerfile
	tar, err := archive.TarWithOptions(dockerfileDir, &archive.TarOptions{})
	if err != nil {
		log.Error("Error creating tar archive", "error", err)
		return err
	}

	targetArch := "arm64"

	opts := types.ImageBuildOptions{
		Dockerfile: "Dockerfile",
		Context:    tar,
		Tags:       []string{DefaultImageName},
		Remove:     true,
		BuildArgs: map[string]*string{
			"TARGETARCH": &targetArch,
		},
	}

	resp, err := b.client.ImageBuild(ctx, tar, opts)
	if err != nil {
		log.Error("Error building image", "error", err)
		return err
	}
	defer resp.Body.Close()

	return b.processAndPrintBuildOutput(resp.Body)
}

func (b *Builder) processAndPrintBuildOutput(reader io.Reader) error {
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

			log.Error("Error decoding build output", "error", err)
			return err
		}

		if message.Error != "" {
			log.Error("Build error", "error", message.Error)
			return fmt.Errorf("build error: %s", message.Error)
		}

		if message.Stream != "" {
			fmt.Print(message.Stream)
		}
	}
}
