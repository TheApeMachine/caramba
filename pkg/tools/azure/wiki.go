package azure

import (
	"context" // Keep json import for potential future use if results need marshaling
	"fmt"

	"github.com/microsoft/azure-devops-go-api/azuredevops/v7"
	"github.com/microsoft/azure-devops-go-api/azuredevops/v7/wiki"
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

// Helper function to get string arguments (consider moving to a shared utils package if used widely)
func getStringArgWiki(args map[string]interface{}, key string) (string, error) {
	val, ok := args[key].(string)
	if !ok {
		return "", fmt.Errorf("missing or invalid type for argument '%s'", key)
	}
	return val, nil
}

/*
CreatePage creates a new wiki page in Azure DevOps.

It uses arguments from the map to set the page content and path.
Returns the page response or an error.
*/
func (w *Wiki) CreatePage(ctx context.Context, args map[string]interface{}) (*wiki.WikiPageResponse, error) {
	project, err := getStringArgWiki(args, "project")
	if err != nil {
		return nil, errnie.Error(err)
	}
	wikiIdentifier, err := getStringArgWiki(args, "wiki_id")
	if err != nil {
		return nil, errnie.Error(err)
	}
	pagePath, err := getStringArgWiki(args, "path")
	if err != nil {
		return nil, errnie.Error(err)
	}
	content, err := getStringArgWiki(args, "content")
	if err != nil {
		return nil, errnie.Error(err)
	}

	pageResponse, err := w.wiki.CreateOrUpdatePage(ctx, wiki.CreateOrUpdatePageArgs{
		Project:        &project,
		WikiIdentifier: &wikiIdentifier,
		Path:           &pagePath,
		Parameters: &wiki.WikiPageCreateOrUpdateParameters{
			Content: &content,
		},
	})

	if err != nil {
		return nil, errnie.Error(err)
	}

	return pageResponse, nil
}

/*
UpdatePage updates an existing wiki page in Azure DevOps.

It uses arguments from the map to update the page content and requires a version.
Returns the page response or an error.
*/
func (w *Wiki) UpdatePage(ctx context.Context, args map[string]interface{}) (*wiki.WikiPageResponse, error) {
	project, err := getStringArgWiki(args, "project")
	if err != nil {
		return nil, errnie.Error(err)
	}
	wikiIdentifier, err := getStringArgWiki(args, "wiki_id")
	if err != nil {
		return nil, errnie.Error(err)
	}
	pagePath, err := getStringArgWiki(args, "path")
	if err != nil {
		return nil, errnie.Error(err)
	}
	content, err := getStringArgWiki(args, "content")
	if err != nil {
		return nil, errnie.Error(err)
	}
	version, err := getStringArgWiki(args, "version") // Version is typically required for updates
	if err != nil {
		return nil, errnie.Error(err)
	}

	pageResponse, err := w.wiki.CreateOrUpdatePage(ctx, wiki.CreateOrUpdatePageArgs{
		Project:        &project,
		WikiIdentifier: &wikiIdentifier,
		Path:           &pagePath,
		Version:        &version,
		Parameters: &wiki.WikiPageCreateOrUpdateParameters{
			Content: &content,
		},
	})

	if err != nil {
		return nil, errnie.Error(err)
	}

	return pageResponse, nil
}

/*
GetPage retrieves a single wiki page from Azure DevOps.

The page path and wiki identifier are extracted from the arguments map.
Returns the page response or an error.
*/
func (w *Wiki) GetPage(ctx context.Context, args map[string]interface{}) (*wiki.WikiPageResponse, error) {
	project, err := getStringArgWiki(args, "project")
	if err != nil {
		return nil, errnie.Error(err)
	}
	wikiIdentifier, err := getStringArgWiki(args, "wiki_id")
	if err != nil {
		return nil, errnie.Error(err)
	}
	pagePath, err := getStringArgWiki(args, "path")
	if err != nil {
		return nil, errnie.Error(err)
	}
	includeContent := true // Typically want content when getting a single page

	pageResponse, err := w.wiki.GetPage(ctx, wiki.GetPageArgs{
		Project:        &project,
		WikiIdentifier: &wikiIdentifier,
		Path:           &pagePath,
		IncludeContent: &includeContent,
	})

	if err != nil {
		return nil, errnie.Error(err)
	}

	return pageResponse, nil
}

/*
ListPages retrieves all wiki pages from a specific wiki in Azure DevOps.

The wiki identifier is extracted from the arguments map.
Returns the batch response value containing the list of pages or an error.
*/
func (w *Wiki) ListPages(ctx context.Context, args map[string]interface{}) (*wiki.GetPagesBatchResponseValue, error) {
	project, err := getStringArgWiki(args, "project")
	if err != nil {
		return nil, errnie.Error(err)
	}
	wikiIdentifier, err := getStringArgWiki(args, "wiki_id")
	if err != nil {
		return nil, errnie.Error(err)
	}

	pagesResponse, err := w.wiki.GetPagesBatch(ctx, wiki.GetPagesBatchArgs{
		Project:        &project,
		WikiIdentifier: &wikiIdentifier,
	})

	if err != nil {
		return nil, errnie.Error(err)
	}

	return pagesResponse, nil
}
