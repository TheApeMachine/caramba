package environment

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/containerd/containerd/v2/client"
	"github.com/google/go-containerregistry/pkg/crane"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/mutate"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/fs"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

/*
Container manages containerd container instances.

It provides functionality for creating, configuring, and managing containers
using the containerd runtime. Each container instance maintains its own client
connection and image reference.
*/
type Container struct {
	conn      *client.Client
	container client.Container
	image     client.Image
}

/*
NewContainer creates a new Container instance.

It initializes a container manager with the provided containerd client connection.
Returns nil if the client connection is invalid.
*/
func NewContainer(conn *client.Client) *Container {
	if conn == nil {
		errnie.Error(fmt.Errorf("client is nil"))
		return nil
	}

	return &Container{
		conn: conn,
	}
}

/*
Load prepares and loads a container image from a Dockerfile.

It performs the following steps:
1. Reads the Dockerfile from the filesystem
2. Pulls the base Ubuntu image with appropriate platform settings
3. Creates a new layer with the Dockerfile contents
4. Builds and imports the final image into containerd
5. Sets up necessary snapshot directories with proper permissions

Returns an error if any step in the process fails.
*/
func (container *Container) Load() (err error) {
	var (
		artifact = datura.New()
		payload  []byte
	)

	ch := make(chan *datura.ArtifactBuilder)

	go func() {
		ch <- artifact
	}()

	artifact = <-fs.NewStore().Generate(ch)

	if payload, err = artifact.DecryptPayload(); err != nil {
		return errnie.Error(err)
	}

	// Create a unique reference for the image
	imageName := "caramba-env:" + time.Now().Format("20060102-150405")

	// Pull base image with platform specification
	platform := &v1.Platform{
		OS:           tweaker.GetOS(),
		Architecture: tweaker.GetArch(),
		Variant:      tweaker.GetVariant(),
	}

	baseImg, err := crane.Pull(
		"ubuntu:latest", // Ubuntu has good ARM64 support
		crane.WithPlatform(platform),
	)

	if err != nil {
		return errnie.Error(err)
	}

	// Create layer with our Dockerfile contents
	files := map[string][]byte{
		"Dockerfile": payload,
	}

	layer, err := crane.Layer(files)

	if err != nil {
		return errnie.Error(err)
	}

	// Append our layer to base image
	newImg, err := mutate.AppendLayers(baseImg, layer)

	if err != nil {
		return errnie.Error(err)
	}

	// Save image to a temporary tarball in OCI format
	tempTar := fmt.Sprintf("/tmp/caramba-env-%s.tar", time.Now().Format("20060102-150405"))

	if err := crane.Save(newImg, imageName, tempTar); err != nil {
		return errnie.Error(err)
	}

	// Read the tarball
	imgBytes, err := os.ReadFile(tempTar)

	if err != nil {
		return errnie.Error(err)
	}

	defer os.Remove(tempTar)

	// Import the image into containerd using OCI format
	images, err := container.conn.Import(
		context.Background(),
		bytes.NewReader(imgBytes),
		client.WithAllPlatforms(true),
		client.WithIndexName(imageName),
	)

	if err != nil {
		return errnie.Error(err)
	}

	if len(images) == 0 {
		return errnie.Error(fmt.Errorf("no images imported"))
	}

	// Get the actual client.Image type
	if container.image, err = container.conn.GetImage(
		context.Background(), images[0].Name,
	); err != nil {
		return errnie.Error(fmt.Errorf("failed to get image: %w", err))
	}

	// Prepare snapshot directories with proper permissions
	snapshotDir := filepath.Join(os.Getenv("CONTAINERD_ROOT"), "snapshots")
	snapshotSubDirs := []string{
		filepath.Join(snapshotDir, "snapshots"),
		filepath.Join(snapshotDir, "metadata"),
		filepath.Join(snapshotDir, "committed"),
		filepath.Join(snapshotDir, "active"),
	}

	// Create all necessary snapshot directories with proper permissions
	for _, dir := range snapshotSubDirs {
		if err := os.MkdirAll(dir, 0777); err != nil {
			return errnie.Error(fmt.Errorf("failed to create snapshot directory %s: %w", dir, err))
		}
		if err := os.Chown(dir, os.Getuid(), os.Getgid()); err != nil {
			return errnie.Error(fmt.Errorf("failed to chown snapshot directory %s: %w", dir, err))
		}
	}

	// Ensure image is unpacked before creating container
	if err = container.image.Unpack(context.Background(), "native"); err != nil {
		return errnie.Error(fmt.Errorf("failed to unpack image: %w", err))
	}

	// Try to load existing container first
	if container.container, err = container.conn.LoadContainer(
		context.Background(), "caramba",
	); err == nil {
		return nil
	}

	// If container doesn't exist, create it
	if container.container, err = container.conn.NewContainer(
		context.Background(),
		"caramba",
		client.WithNewSnapshot("caramba-snap", container.image),
		client.WithRuntime("io.containerd.runc.v2", nil),
		client.WithSnapshotter("native"),
	); err != nil {
		return errnie.Error(fmt.Errorf("failed to create container: %w", err))
	}

	// Double check snapshot permissions after container creation
	snapshotDir = filepath.Join(os.Getenv("CONTAINERD_ROOT"), "snapshots")
	walkErr := filepath.Walk(snapshotDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if err := os.Chmod(path, 0777); err != nil {
			return fmt.Errorf("failed to chmod %s: %w", path, err)
		}
		if err := os.Chown(path, os.Getuid(), os.Getgid()); err != nil {
			return fmt.Errorf("failed to chown %s: %w", path, err)
		}
		return nil
	})
	if walkErr != nil {
		return errnie.Error(fmt.Errorf("failed to set snapshot permissions: %w", walkErr))
	}

	return nil
}
