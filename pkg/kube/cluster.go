package kube

import (
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
	"sigs.k8s.io/kind/pkg/cluster"
)

type Cluster struct {
	buffer   *stream.Buffer
	provider *cluster.Provider
}

func NewCluster() *Cluster {
	errnie.Debug("kube.NewCluster")
	prvdr := cluster.NewProvider()

	return &Cluster{
		buffer: stream.NewBuffer(func(evt *datura.Artifact) (err error) {
			errnie.Debug("kube.Cluster.buffer.fn")

			prvdr.Create("caramba")

			return nil
		}),
		provider: prvdr,
	}
}

func (prvdr *Cluster) Read(p []byte) (n int, err error) {
	errnie.Debug("kube.Cluster.Read")
	return prvdr.buffer.Read(p)
}

func (prvdr *Cluster) Write(p []byte) (n int, err error) {
	errnie.Debug("kube.Cluster.Write")
	return prvdr.buffer.Write(p)
}

func (prvdr *Cluster) Close() error {
	errnie.Debug("kube.Cluster.Close")
	return prvdr.buffer.Close()
}
