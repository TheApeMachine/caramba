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
	prvdr := cluster.NewProvider()
	store := fs.NewStore()
	deployment := NewDeployment(store)

	return &Cluster{
		buffer: stream.NewBuffer(func(evt *datura.Artifact) (err error) {
			errnie.Debug("kube.Cluster.buffer.fn")

			// Create cluster with custom kubeconfig path
			if err = prvdr.Create(
				"caramba",
				cluster.CreateWithKubeconfigPath(filepath.Join(carambaKubeDir, "config")),
			); err != nil {
				return errnie.Error(err)
			}

			// After cluster creation, deploy manifests
			if err = deployment.Apply(); err != nil {
				return errnie.Error(err)
			}

			return nil
		}),
		provider:   prvdr,
		store:      store,
		deployment: deployment,
	}
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
