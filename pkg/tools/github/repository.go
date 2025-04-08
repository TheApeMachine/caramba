package github

import (
	"bytes"
	"context"
	"encoding/json"

	"github.com/google/go-github/v70/github"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Repository manages GitHub repository operations through the GitHub API.
It provides functionality to create, query, and manage repository information
including branches and file contents.
*/
type Repository struct {
	conn *github.Client
}

/*
NewRepository creates a new Repository instance with the provided GitHub client connection.
*/
func NewRepository(conn *github.Client) *Repository {
	return &Repository{conn: conn}
}

/*
encode serializes the provided value into JSON and adds it to the artifact's payload.

Returns an error if JSON encoding fails.
*/
func (repository *Repository) encode(artifact datura.Artifact, v any) (err error) {
	payload := bytes.NewBuffer([]byte{})

	if err = json.NewEncoder(payload).Encode(v); err != nil {
		return errnie.Error(err)
	}

	datura.WithEncryptedPayload(payload.Bytes())(artifact)

	return nil
}

/*
GetRepositories retrieves all repositories accessible to the authenticated user.

Returns an error if the retrieval fails.
*/
func (repository *Repository) GetRepositories(artifact datura.Artifact) (err error) {
	repos, _, err := repository.conn.Repositories.ListByAuthenticatedUser(
		context.Background(),
		nil,
	)

	if err != nil {
		return errnie.Error(err)
	}

	return repository.encode(artifact, repos)
}

/*
GetRepository retrieves information about a specific repository.

Uses owner and repository name from the artifact's metadata.
Returns an error if the retrieval fails.
*/
func (repository *Repository) GetRepository(artifact datura.Artifact) (err error) {
	repo, _, err := repository.conn.Repositories.Get(
		context.Background(),
		datura.GetMetaValue[string](artifact, "owner"),
		datura.GetMetaValue[string](artifact, "name"),
	)

	if err != nil {
		return errnie.Error(err)
	}

	return repository.encode(artifact, repo)
}

/*
CreateRepository creates a new repository for the authenticated user.

Uses metadata from the artifact to set repository fields like name,
description, and visibility. Returns an error if the creation fails.
*/
func (repository *Repository) CreateRepository(artifact datura.Artifact) (err error) {
	repo := &github.Repository{
		Name:        github.Ptr(datura.GetMetaValue[string](artifact, "name")),
		Description: github.Ptr(datura.GetMetaValue[string](artifact, "description")),
		Private:     github.Ptr(datura.GetMetaValue[bool](artifact, "private")),
		AutoInit:    github.Ptr(true),
	}

	created, _, err := repository.conn.Repositories.Create(
		context.Background(),
		"",
		repo,
	)
	if err != nil {
		return errnie.Error(err)
	}
	return repository.encode(artifact, created)
}

/*
ListBranches retrieves all branches from a repository.

Uses owner and repository name from the artifact's metadata.
Returns an error if the retrieval fails.
*/
func (repository *Repository) ListBranches(artifact datura.Artifact) (err error) {
	branches, _, err := repository.conn.Repositories.ListBranches(
		context.Background(),
		datura.GetMetaValue[string](artifact, "owner"),
		datura.GetMetaValue[string](artifact, "name"),
		nil,
	)
	if err != nil {
		return errnie.Error(err)
	}
	return repository.encode(artifact, branches)
}

/*
GetContents retrieves the contents of a file or directory in a repository.

Uses owner, repository name, and file path from the artifact's metadata.
Returns an error if the retrieval fails.
*/
func (repository *Repository) GetContents(artifact datura.Artifact) (err error) {
	content, _, _, err := repository.conn.Repositories.GetContents(
		context.Background(),
		datura.GetMetaValue[string](artifact, "owner"),
		datura.GetMetaValue[string](artifact, "name"),
		datura.GetMetaValue[string](artifact, "path"),
		nil,
	)
	if err != nil {
		return errnie.Error(err)
	}
	return repository.encode(artifact, content)
}
