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

type Wiki struct {
	conn *azuredevops.Connection
	wiki wiki.Client
}

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

func (w *Wiki) encode(artifact *datura.Artifact, v any) (err error) {
	payload := bytes.NewBuffer([]byte{})

	if err = json.NewEncoder(payload).Encode(v); err != nil {
		return errnie.Error(err)
	}

	datura.WithPayload(payload.Bytes())(artifact)
	return nil
}

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
