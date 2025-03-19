package kube

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/fs"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/yaml"
)

type Deployment struct {
	store  *fs.Store
	config *rest.Config
	client dynamic.Interface
	mapper *restmapper.DeferredDiscoveryRESTMapper
}

func NewDeployment(store *fs.Store) *Deployment {
	config, err := rest.InClusterConfig()

	if err != nil {
		home, err := os.UserHomeDir()

		if err != nil {
			errnie.Error(err)
			return nil
		}

		kubeconfig := filepath.Join(home, ".caramba", "kube", "config")
		config, err = clientcmd.BuildConfigFromFlags("", kubeconfig)

		if err != nil {
			errnie.Error(err)
			return nil
		}
	}

	// Create the dynamic client
	client, err := dynamic.NewForConfig(config)
	if err != nil {
		errnie.Error(err)
		return nil
	}

	// Create the discovery client and mapper
	dc, err := discovery.NewDiscoveryClientForConfig(config)
	if err != nil {
		errnie.Error(err)
		return nil
	}
	mapper := restmapper.NewDeferredDiscoveryRESTMapper(memory.NewMemCacheClient(dc))

	return &Deployment{
		store:  store,
		config: config,
		client: client,
		mapper: mapper,
	}
}

func (deployment *Deployment) Apply() (err error) {
	errnie.Debug("kube.Deployment.Apply")

	// Create an artifact to list files
	listArtifact := datura.New(
		datura.WithRole(datura.ArtifactRoleListFiles),
		datura.WithMeta("path", "manifests"),
	)

	if _, err = io.Copy(deployment.store, listArtifact); err != nil {
		return errnie.Error(err)
	}

	buf := bytes.NewBuffer([]byte{})

	// Read the response
	if _, err = io.Copy(buf, deployment.store); err != nil {
		return errnie.Error(err)
	}

	// Process the list of files
	files := bytes.Split(buf.Bytes(), []byte("\n"))
	for _, file := range files {
		if len(file) == 0 {
			continue
		}

		// Create an artifact to read each manifest
		readArtifact := datura.New(
			datura.WithRole(datura.ArtifactRoleOpenFile),
			datura.WithMeta("path", filepath.Join("manifests", string(file))),
		)

		// Write the read request to the store
		if _, err = io.Copy(deployment.store, readArtifact); err != nil {
			return errnie.Error(err)
		}

		// Read the manifest content
		manifestBuf := bytes.NewBuffer([]byte{})
		if _, err = io.Copy(manifestBuf, deployment.store); err != nil {
			return errnie.Error(err)
		}

		// Convert YAML to JSON and decode into unstructured object
		obj := &unstructured.Unstructured{}
		if err = yaml.Unmarshal(manifestBuf.Bytes(), obj); err != nil {
			return errnie.Error(err)
		}

		// Find the GVR for this object
		gvk := obj.GroupVersionKind()
		mapping, err := deployment.mapper.RESTMapping(gvk.GroupKind(), gvk.Version)
		if err != nil {
			return errnie.Error(err)
		}

		// Prepare the dynamic resource interface
		var dr dynamic.ResourceInterface
		if mapping.Scope.Name() == meta.RESTScopeNameNamespace {
			dr = deployment.client.Resource(mapping.Resource).Namespace(obj.GetNamespace())
		} else {
			dr = deployment.client.Resource(mapping.Resource)
		}

		// Apply the manifest using server-side apply
		_, err = dr.Apply(context.Background(), obj.GetName(), obj, metav1.ApplyOptions{
			FieldManager: "caramba-controller",
			Force:        true,
		})
		if err != nil {
			return errnie.Error(err)
		}
	}

	return nil
}
