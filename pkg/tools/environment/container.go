package environment

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/containerd/containerd/v2/client"
	"github.com/google/go-containerregistry/pkg/crane"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/mutate"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/fs"
	"github.com/theapemachine/caramba/pkg/workflow"
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
	if conn == nil {
		errnie.Error(fmt.Errorf("client is nil"))
		return nil
	}

	return &Container{
		conn: conn,
	}
}

/*
Load prepares and loads a new container image from a Dockerfile.
It creates a build context, imports the image into containerd, and initializes a new container.
*/
func (container *Container) Load() (err error) {
	var (
		artifact = datura.New()
		filesys  = fs.NewStore()
		payload  []byte
	)

	// Read our Dockerfile
	if _, err = io.Copy(artifact, workflow.NewPipeline(datura.New(
		datura.WithRole(datura.ArtifactRoleOpenFile),
		datura.WithMeta("path", "manifests/Dockerfile"),
	), filesys)); err != nil {
		return errnie.Error(err)
	}

	if payload, err = artifact.DecryptPayload(); errnie.Error(err) != nil {
		return err
	}

	// Create a unique reference for the image
	imageName := "caramba-env:" + time.Now().Format("20060102-150405")

	// Pull base image with platform specification
	platform := &v1.Platform{
		OS:           "linux",
		Architecture: "arm64",
		Variant:      "v8", // Common for M1/M2 Macs
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
	container.image, err = container.conn.GetImage(context.Background(), images[0].Name)
	if err != nil {
		return errnie.Error(err)
	}

	// Ensure image is unpacked before creating container
	if err := container.image.Unpack(context.Background(), ""); err != nil {
		return errnie.Error(err)
	}

	// Create the container with a unique snapshot ID
	snapshotID := fmt.Sprintf("caramba-snap-%s", time.Now().Format("20060102-150405"))
	container.container, err = container.conn.NewContainer(
		context.Background(),
		"caramba",
		client.WithNewSnapshot(snapshotID, container.image),
		client.WithRuntime("io.containerd.runtime.v1.linux", nil),
	)

	return errnie.Error(err)
}
