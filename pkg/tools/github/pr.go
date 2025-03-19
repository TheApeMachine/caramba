package github

import (
	"bytes"
	"context"
	"encoding/json"

	"github.com/google/go-github/v70/github"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type PR struct {
	conn *github.Client
}

func NewPR(conn *github.Client) *PR {
	return &PR{conn: conn}
}

func (pr *PR) encode(artifact *datura.Artifact, v any) (err error) {
	payload := bytes.NewBuffer([]byte{})

	if err = json.NewEncoder(payload).Encode(v); err != nil {
		return errnie.Error(err)
	}

	datura.WithPayload(payload.Bytes())(artifact)

	return nil
}

func (pr *PR) GetPR(artifact *datura.Artifact) (err error) {
	pullRequest, _, err := pr.conn.PullRequests.Get(
		context.Background(),
		datura.GetMetaValue[string](artifact, "owner"),
		datura.GetMetaValue[string](artifact, "name"),
		datura.GetMetaValue[int](artifact, "number"),
	)
	if err != nil {
		return errnie.Error(err)
	}

	return pr.encode(artifact, pullRequest)
}

func (pr *PR) ListPRs(artifact *datura.Artifact) (err error) {
	pullRequests, _, err := pr.conn.PullRequests.List(
		context.Background(),
		datura.GetMetaValue[string](artifact, "owner"),
		datura.GetMetaValue[string](artifact, "name"),
		nil,
	)
	if err != nil {
		return errnie.Error(err)
	}
	return pr.encode(artifact, pullRequests)
}

func (pr *PR) CreatePR(artifact *datura.Artifact) (err error) {
	newPR := &github.NewPullRequest{
		Title:               github.String(datura.GetMetaValue[string](artifact, "title")),
		Head:                github.String(datura.GetMetaValue[string](artifact, "head")),
		Base:                github.String(datura.GetMetaValue[string](artifact, "base")),
		Body:                github.String(datura.GetMetaValue[string](artifact, "body")),
		MaintainerCanModify: github.Bool(true),
	}

	pullRequest, _, err := pr.conn.PullRequests.Create(
		context.Background(),
		datura.GetMetaValue[string](artifact, "owner"),
		datura.GetMetaValue[string](artifact, "name"),
		newPR,
	)
	if err != nil {
		return errnie.Error(err)
	}
	return pr.encode(artifact, pullRequest)
}

func (pr *PR) UpdatePR(artifact *datura.Artifact) (err error) {
	update := &github.PullRequest{
		Title: github.String(datura.GetMetaValue[string](artifact, "title")),
		Body:  github.String(datura.GetMetaValue[string](artifact, "body")),
		State: github.String(datura.GetMetaValue[string](artifact, "state")),
	}

	pullRequest, _, err := pr.conn.PullRequests.Edit(
		context.Background(),
		datura.GetMetaValue[string](artifact, "owner"),
		datura.GetMetaValue[string](artifact, "name"),
		datura.GetMetaValue[int](artifact, "number"),
		update,
	)
	if err != nil {
		return errnie.Error(err)
	}
	return pr.encode(artifact, pullRequest)
}
