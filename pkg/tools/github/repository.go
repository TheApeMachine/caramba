package github

import (
	"bytes"
	"context"
	"encoding/json"

	"github.com/google/go-github/v70/github"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type Repository struct {
	conn *github.Client
}

func NewRepository(conn *github.Client) *Repository {
	return &Repository{conn: conn}
}

func (repository *Repository) encode(artifact *datura.Artifact, v any) (err error) {
	payload := bytes.NewBuffer([]byte{})

	if err = json.NewEncoder(payload).Encode(v); err != nil {
		return errnie.Error(err)
	}

	datura.WithPayload(payload.Bytes())(artifact)

	return nil
}

func (repository *Repository) GetRepositories(artifact *datura.Artifact) (err error) {
	repos, _, err := repository.conn.Repositories.ListByAuthenticatedUser(
		context.Background(),
		nil,
	)

	if err != nil {
		return errnie.Error(err)
	}

	return repository.encode(artifact, repos)
}

func (repository *Repository) GetRepository(artifact *datura.Artifact) (err error) {
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

func (repository *Repository) CreateRepository(artifact *datura.Artifact) (err error) {
	repo := &github.Repository{
		Name:        github.String(datura.GetMetaValue[string](artifact, "name")),
		Description: github.String(datura.GetMetaValue[string](artifact, "description")),
		Private:     github.Bool(datura.GetMetaValue[bool](artifact, "private")),
		AutoInit:    github.Bool(true),
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

func (repository *Repository) ListBranches(artifact *datura.Artifact) (err error) {
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

func (repository *Repository) GetContents(artifact *datura.Artifact) (err error) {
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
