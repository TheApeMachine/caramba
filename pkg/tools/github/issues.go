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
Issues manages GitHub issue operations through the GitHub API.
It provides functionality to create, update, and query repository issues.
*/
type Issues struct {
	conn *github.Client
}

/*
NewIssues creates a new Issues instance with the provided GitHub client connection.
*/
func NewIssues(conn *github.Client) *Issues {
	return &Issues{conn: conn}
}

/*
encode serializes the provided value into JSON and adds it to the artifact's payload.

Returns an error if JSON encoding fails.
*/
func (issues *Issues) encode(artifact *datura.ArtifactBuilder, v any) (err error) {
	payload := bytes.NewBuffer([]byte{})
	if err = json.NewEncoder(payload).Encode(v); err != nil {
		return errnie.Error(err)
	}
	datura.WithEncryptedPayload(payload.Bytes())(artifact)
	return nil
}

/*
GetIssue retrieves a single issue from a repository.

Uses owner, repository name, and issue number from the artifact's metadata.
Returns an error if the retrieval fails.
*/
func (issues *Issues) GetIssue(artifact *datura.ArtifactBuilder) (err error) {
	issue, _, err := issues.conn.Issues.Get(
		context.Background(),
		datura.GetMetaValue[string](artifact, "owner"),
		datura.GetMetaValue[string](artifact, "name"),
		datura.GetMetaValue[int](artifact, "number"),
	)
	if err != nil {
		return errnie.Error(err)
	}
	return issues.encode(artifact, issue)
}

/*
ListIssues retrieves all issues from a repository.

Uses owner and repository name from the artifact's metadata.
Returns an error if the retrieval fails.
*/
func (issues *Issues) ListIssues(artifact *datura.ArtifactBuilder) (err error) {
	issueList, _, err := issues.conn.Issues.ListByRepo(
		context.Background(),
		datura.GetMetaValue[string](artifact, "owner"),
		datura.GetMetaValue[string](artifact, "name"),
		nil,
	)
	if err != nil {
		return errnie.Error(err)
	}
	return issues.encode(artifact, issueList)
}

/*
CreateIssue creates a new issue in a repository.

Uses metadata from the artifact to set issue fields like title and body.
Returns an error if the creation fails.
*/
func (issues *Issues) CreateIssue(artifact *datura.ArtifactBuilder) (err error) {
	issue := &github.IssueRequest{
		Title:     github.Ptr(datura.GetMetaValue[string](artifact, "title")),
		Body:      github.Ptr(datura.GetMetaValue[string](artifact, "body")),
		Labels:    &[]string{},
		Assignees: &[]string{},
	}

	created, _, err := issues.conn.Issues.Create(
		context.Background(),
		datura.GetMetaValue[string](artifact, "owner"),
		datura.GetMetaValue[string](artifact, "name"),
		issue,
	)
	if err != nil {
		return errnie.Error(err)
	}
	return issues.encode(artifact, created)
}

/*
UpdateIssue updates an existing issue in a repository.

Uses metadata from the artifact to update issue fields like title, body, and state.
Returns an error if the update fails.
*/
func (issues *Issues) UpdateIssue(artifact *datura.ArtifactBuilder) (err error) {
	update := &github.IssueRequest{
		Title: github.Ptr(datura.GetMetaValue[string](artifact, "title")),
		Body:  github.Ptr(datura.GetMetaValue[string](artifact, "body")),
		State: github.Ptr(datura.GetMetaValue[string](artifact, "state")),
	}

	updated, _, err := issues.conn.Issues.Edit(
		context.Background(),
		datura.GetMetaValue[string](artifact, "owner"),
		datura.GetMetaValue[string](artifact, "name"),
		datura.GetMetaValue[int](artifact, "number"),
		update,
	)
	if err != nil {
		return errnie.Error(err)
	}
	return issues.encode(artifact, updated)
}
