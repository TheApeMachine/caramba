package kube

import (
	"os"
	"path/filepath"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/fs"
	"github.com/theapemachine/caramba/pkg/stream"
	"k8s.io/client-go/util/homedir"
	"sigs.k8s.io/kind/pkg/cluster"
)

type Cluster struct {
	buffer     *stream.Buffer
	provider   *cluster.Provider
	store      *fs.Store
	deployment *Deployment
}

func NewCluster() *Cluster {
	errnie.Debug("kube.NewCluster")

	// Setup caramba kube directory
	home := homedir.HomeDir()
	carambaKubeDir := filepath.Join(home, ".caramba", "kube")
	if err := os.MkdirAll(carambaKubeDir, 0755); err != nil {
		errnie.Error(err)
		return nil
	}

	// Configure kind cluster with custom kubeconfig
	prvdr := cluster.NewProvider(cluster.ProviderWithLogger(NewKindLogger()))
	store := fs.NewStore()
	deployment := NewDeployment(store)

	return &Cluster{
		buffer:     stream.NewBuffer(),
		provider:   prvdr,
		store:      store,
		deployment: deployment,
	}
}

func (c *Cluster) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("kube.Cluster.Generate")

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		select {
		case <-buffer:
			// Setup caramba kube directory
			home := homedir.HomeDir()
			carambaKubeDir := filepath.Join(home, ".caramba", "kube")
			kubeConfigPath := filepath.Join(carambaKubeDir, "config")

			// Create cluster with custom kubeconfig path
			if err := c.provider.Create(
				"caramba",
				cluster.CreateWithKubeconfigPath(kubeConfigPath),
			); err != nil {
				out <- datura.New(datura.WithError(errnie.Error(err)))
				return
			}

			// After cluster creation, deploy manifests
			if err := c.deployment.Apply(); err != nil {
				out <- datura.New(datura.WithError(errnie.Error(err)))
				return
			}

			// Return success message
			out <- datura.New(datura.WithPayload([]byte("Kubernetes cluster created and deployed successfully")))
		}
	}()

	return out
}

func (c *Cluster) Read(p []byte) (n int, err error) {
	errnie.Debug("kube.Cluster.Read")
	return c.buffer.Read(p)
}

func (c *Cluster) Write(p []byte) (n int, err error) {
	errnie.Debug("kube.Cluster.Write")
	return c.buffer.Write(p)
}

func (c *Cluster) Close() error {
	errnie.Debug("kube.Cluster.Close")
	return c.buffer.Close()
}
