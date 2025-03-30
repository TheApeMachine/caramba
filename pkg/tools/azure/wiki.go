package azure

import (
	"bytes"
	"context"
	"encoding/json"

	"github.com/microsoft/azure-devops-go-api/azuredevops/v7"
	"github.com/microsoft/azure-devops-go-api/azuredevops/v7/wiki"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Wiki manages Azure DevOps wiki pages through the Azure DevOps API.
It provides functionality to create, update, and query wiki pages.
*/
type Wiki struct {
	conn *azuredevops.Connection
	wiki wiki.Client
}

/*
NewWiki creates a new Wiki instance with the provided Azure DevOps connection.

It initializes the wiki client for interacting with wiki pages.
Returns nil if client initialization fails.
*/
func NewWiki(conn *azuredevops.Connection) *Wiki {
	ctx := context.Background()
	wikiClient, err := wiki.NewClient(ctx, conn)

	if err != nil {
		errnie.Error(err)
		return nil
	}

	return &Wiki{
		conn: conn,
		wiki: wikiClient,
	}
}

/*
encode serializes the provided value into JSON and adds it to the artifact's payload.

Returns an error if JSON encoding fails.
*/
func (w *Wiki) encode(artifact *datura.Artifact, v any) (err error) {
	payload := bytes.NewBuffer([]byte{})

	if err = json.NewEncoder(payload).Encode(v); err != nil {
		return errnie.Error(err)
	}

	datura.WithPayload(payload.Bytes())(artifact)
	return nil
}

/*
CreatePage creates a new wiki page in Azure DevOps.

It uses metadata from the artifact to set the page content and path.
Returns an error if the creation fails.
*/
func (w *Wiki) CreatePage(artifact *datura.Artifact) (err error) {
	ctx := context.Background()
	project := datura.GetMetaValue[string](artifact, "project")
	wikiIdentifier := datura.GetMetaValue[string](artifact, "wiki_id")
	pagePath := datura.GetMetaValue[string](artifact, "path")
	content := datura.GetMetaValue[string](artifact, "content")

	page, err := w.wiki.CreateOrUpdatePage(ctx, wiki.CreateOrUpdatePageArgs{
		Project:        &project,
		WikiIdentifier: &wikiIdentifier,
		Path:           &pagePath,
		Parameters: &wiki.WikiPageCreateOrUpdateParameters{
			Content: &content,
		},
	})

	if err != nil {
		return errnie.Error(err)
	}

	return w.encode(artifact, page)
}

/*
UpdatePage updates an existing wiki page in Azure DevOps.

It uses metadata from the artifact to update the page content and requires a version.
Returns an error if the update fails.
*/
func (w *Wiki) UpdatePage(artifact *datura.Artifact) (err error) {
	ctx := context.Background()
	project := datura.GetMetaValue[string](artifact, "project")
	wikiIdentifier := datura.GetMetaValue[string](artifact, "wiki_id")
	pagePath := datura.GetMetaValue[string](artifact, "path")
	content := datura.GetMetaValue[string](artifact, "content")
	version := datura.GetMetaValue[string](artifact, "version")

	page, err := w.wiki.CreateOrUpdatePage(ctx, wiki.CreateOrUpdatePageArgs{
		Project:        &project,
		WikiIdentifier: &wikiIdentifier,
		Path:           &pagePath,
		Version:        &version,
		Parameters: &wiki.WikiPageCreateOrUpdateParameters{
			Content: &content,
		},
	})

	if err != nil {
		return errnie.Error(err)
	}

	return w.encode(artifact, page)
}

/*
GetPage retrieves a single wiki page from Azure DevOps.

The page path and wiki identifier are extracted from the artifact's metadata.
Returns an error if the retrieval fails.
*/
func (w *Wiki) GetPage(artifact *datura.Artifact) (err error) {
	ctx := context.Background()
	project := datura.GetMetaValue[string](artifact, "project")
	wikiIdentifier := datura.GetMetaValue[string](artifact, "wiki_id")
	pagePath := datura.GetMetaValue[string](artifact, "path")
	includeContent := true

	page, err := w.wiki.GetPage(ctx, wiki.GetPageArgs{
		Project:        &project,
		WikiIdentifier: &wikiIdentifier,
		Path:           &pagePath,
		IncludeContent: &includeContent,
	})

	if err != nil {
		return errnie.Error(err)
	}

	return w.encode(artifact, page)
}

/*
ListPages retrieves all wiki pages from a specific wiki in Azure DevOps.

The wiki identifier is extracted from the artifact's metadata.
Returns an error if the retrieval fails.
*/
func (w *Wiki) ListPages(artifact *datura.Artifact) (err error) {
	ctx := context.Background()
	project := datura.GetMetaValue[string](artifact, "project")
	wikiIdentifier := datura.GetMetaValue[string](artifact, "wiki_id")

	pages, err := w.wiki.GetPagesBatch(ctx, wiki.GetPagesBatchArgs{
		Project:        &project,
		WikiIdentifier: &wikiIdentifier,
	})

	if err != nil {
		return errnie.Error(err)
	}

	return w.encode(artifact, pages)
}
