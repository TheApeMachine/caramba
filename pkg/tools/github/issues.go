package github

import (
	"bytes"
	"context"
	"encoding/json"

	"github.com/google/go-github/v70/github"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type Issues struct {
	conn *github.Client
}

func NewIssues(conn *github.Client) *Issues {
	return &Issues{conn: conn}
}

func (issues *Issues) encode(artifact *datura.Artifact, v any) (err error) {
	payload := bytes.NewBuffer([]byte{})
	if err = json.NewEncoder(payload).Encode(v); err != nil {
		return errnie.Error(err)
	}
	datura.WithPayload(payload.Bytes())(artifact)
	return nil
}

func (issues *Issues) GetIssue(artifact *datura.Artifact) (err error) {
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

func (issues *Issues) ListIssues(artifact *datura.Artifact) (err error) {
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

func (issues *Issues) CreateIssue(artifact *datura.Artifact) (err error) {
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

func (issues *Issues) UpdateIssue(artifact *datura.Artifact) (err error) {
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
