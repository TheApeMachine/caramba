package environment

import (
	"archive/tar"
	"bytes"
	"context"
	"fmt"
	"os"
	"time"

	"github.com/containerd/containerd/v2/client"
	"github.com/spf13/afero"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Container represents a containerd container instance with its associated client, container, and image.
*/
type Container struct {
	conn      *client.Client
	container client.Container
	image     client.Image
	err       error
}

/*
NewContainer creates a new Container instance with the given containerd client connection.
*/
func NewContainer(conn *client.Client) *Container {
	return &Container{
		conn: conn,
	}
}

/*
Load prepares and loads a new container image from a Dockerfile.
It creates a build context, imports the image into containerd, and initializes a new container.
*/
func (container *Container) Load() (err error) {
	// Read the Dockerfile
	dockerfileContent, err := os.ReadFile("pkg/tools/environment/Dockerfile")
	if errnie.Error(err) != nil {
		return err
	}

	// Create a unique reference for the image
	imageName := "caramba-env:" + time.Now().Format("20060102-150405")

	// Use Afero's in-memory filesystem for all temporary operations
	memFs := afero.NewMemMapFs()

	// Create build context directory in memory
	if err := memFs.MkdirAll("/build", 0755); errnie.Error(err) != nil {
		return err
	}

	// Write Dockerfile to memory filesystem
	dockerfileInMem, err := memFs.Create("/build/Dockerfile")
	if errnie.Error(err) != nil {
		return err
	}

	if _, err := dockerfileInMem.Write(dockerfileContent); errnie.Error(err) != nil {
		dockerfileInMem.Close()
		return err
	}
	dockerfileInMem.Close()

	// Create tar buffer in memory
	var tarBuffer bytes.Buffer
	tarWriter := tar.NewWriter(&tarBuffer)

	// Get file info from memory filesystem
	fileInfo, err := memFs.Stat("/build/Dockerfile")
	if errnie.Error(err) != nil {
		return err
	}

	// Create tar header
	header, err := tar.FileInfoHeader(fileInfo, "")
	if errnie.Error(err) != nil {
		return err
	}
	header.Name = "Dockerfile"

	if err := tarWriter.WriteHeader(header); errnie.Error(err) != nil {
		return err
	}

	// Read file from memory filesystem and write to tar
	dockerfileData, err := afero.ReadFile(memFs, "/build/Dockerfile")
	if errnie.Error(err) != nil {
		return err
	}

	if _, err := tarWriter.Write(dockerfileData); errnie.Error(err) != nil {
		return err
	}

	tarWriter.Close()

	// Import the tar buffer directly into containerd
	images, err := container.conn.Import(
		context.Background(),
		bytes.NewReader(tarBuffer.Bytes()),
		client.WithIndexName(imageName),
	)
	if errnie.Error(err) != nil {
		return err
	}

	if len(images) == 0 {
		return errnie.Error(fmt.Errorf("no images imported"))
	}

	// Get the imported image
	container.image, container.err = container.conn.GetImage(context.Background(), images[0].Name)
	if errnie.Error(err) != nil {
		return err
	}

	container.container, container.err = container.conn.NewContainer(
		context.Background(),
		"caramba",
		client.WithNewSnapshot("caramba", container.image),
	)

	return container.err
}
