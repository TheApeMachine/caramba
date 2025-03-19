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
		Title:               github.Ptr(datura.GetMetaValue[string](artifact, "title")),
		Head:                github.Ptr(datura.GetMetaValue[string](artifact, "head")),
		Base:                github.Ptr(datura.GetMetaValue[string](artifact, "base")),
		Body:                github.Ptr(datura.GetMetaValue[string](artifact, "body")),
		MaintainerCanModify: github.Ptr(true),
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
		Title: github.Ptr(datura.GetMetaValue[string](artifact, "title")),
		Body:  github.Ptr(datura.GetMetaValue[string](artifact, "body")),
		State: github.Ptr(datura.GetMetaValue[string](artifact, "state")),
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

func (pr *PR) CreatePRComment(artifact *datura.Artifact) (err error) {
	comment := &github.IssueComment{
		Body: github.Ptr(datura.GetMetaValue[string](artifact, "body")),
	}

	created, _, err := pr.conn.Issues.CreateComment(
		context.Background(),
		datura.GetMetaValue[string](artifact, "owner"),
		datura.GetMetaValue[string](artifact, "name"),
		datura.GetMetaValue[int](artifact, "number"),
		comment,
	)
	if err != nil {
		return errnie.Error(err)
	}
	return pr.encode(artifact, created)
}

func (pr *PR) ListPRComments(artifact *datura.Artifact) (err error) {
	comments, _, err := pr.conn.Issues.ListComments(
		context.Background(),
		datura.GetMetaValue[string](artifact, "owner"),
		datura.GetMetaValue[string](artifact, "name"),
		datura.GetMetaValue[int](artifact, "number"),
		nil,
	)
	if err != nil {
		return errnie.Error(err)
	}
	return pr.encode(artifact, comments)
}

func (pr *PR) CreateReviewComment(artifact *datura.Artifact) (err error) {
	comment := &github.PullRequestComment{
		Body:     github.Ptr(datura.GetMetaValue[string](artifact, "body")),
		Path:     github.Ptr(datura.GetMetaValue[string](artifact, "path")),
		Position: github.Ptr(datura.GetMetaValue[int](artifact, "position")),
	}

	created, _, err := pr.conn.PullRequests.CreateComment(
		context.Background(),
		datura.GetMetaValue[string](artifact, "owner"),
		datura.GetMetaValue[string](artifact, "name"),
		datura.GetMetaValue[int](artifact, "number"),
		comment,
	)
	if err != nil {
		return errnie.Error(err)
	}
	return pr.encode(artifact, created)
}

func (pr *PR) ListReviewComments(artifact *datura.Artifact) (err error) {
	comments, _, err := pr.conn.PullRequests.ListComments(
		context.Background(),
		datura.GetMetaValue[string](artifact, "owner"),
		datura.GetMetaValue[string](artifact, "name"),
		datura.GetMetaValue[int](artifact, "number"),
		nil,
	)
	if err != nil {
		return errnie.Error(err)
	}
	return pr.encode(artifact, comments)
}

func (pr *PR) CreatePRReview(artifact *datura.Artifact) (err error) {
	review := &github.PullRequestReviewRequest{
		Body:  github.Ptr(datura.GetMetaValue[string](artifact, "body")),
		Event: github.Ptr(datura.GetMetaValue[string](artifact, "event")), // APPROVE, REQUEST_CHANGES, COMMENT
		Comments: []*github.DraftReviewComment{
			{
				Path:     github.Ptr(datura.GetMetaValue[string](artifact, "path")),
				Position: github.Ptr(datura.GetMetaValue[int](artifact, "position")),
				Body:     github.Ptr(datura.GetMetaValue[string](artifact, "comment")),
			},
		},
	}

	created, _, err := pr.conn.PullRequests.CreateReview(
		context.Background(),
		datura.GetMetaValue[string](artifact, "owner"),
		datura.GetMetaValue[string](artifact, "name"),
		datura.GetMetaValue[int](artifact, "number"),
		review,
	)
	if err != nil {
		return errnie.Error(err)
	}
	return pr.encode(artifact, created)
}

func (pr *PR) ListPRReviews(artifact *datura.Artifact) (err error) {
	reviews, _, err := pr.conn.PullRequests.ListReviews(
		context.Background(),
		datura.GetMetaValue[string](artifact, "owner"),
		datura.GetMetaValue[string](artifact, "name"),
		datura.GetMetaValue[int](artifact, "number"),
		nil,
	)
	if err != nil {
		return errnie.Error(err)
	}
	return pr.encode(artifact, reviews)
}

func (pr *PR) SubmitReview(artifact *datura.Artifact) (err error) {
	review := &github.PullRequestReviewRequest{
		Body:  github.Ptr(datura.GetMetaValue[string](artifact, "body")),
		Event: github.Ptr(datura.GetMetaValue[string](artifact, "event")), // APPROVE, REQUEST_CHANGES, COMMENT
	}

	submitted, _, err := pr.conn.PullRequests.SubmitReview(
		context.Background(),
		datura.GetMetaValue[string](artifact, "owner"),
		datura.GetMetaValue[string](artifact, "name"),
		datura.GetMetaValue[int](artifact, "number"),
		int64(datura.GetMetaValue[int](artifact, "review_id")),
		review,
	)
	if err != nil {
		return errnie.Error(err)
	}
	return pr.encode(artifact, submitted)
}
