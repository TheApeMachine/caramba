package github

import (
	"context"

	"github.com/google/go-github/v70/github"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
PR manages GitHub pull request operations through the GitHub API.
It provides functionality to create, update, review, and manage pull requests
and their associated comments and reviews.
*/
type PR struct {
	conn *github.Client
}

/*
NewPR creates a new PR instance with the provided GitHub client connection.
*/
func NewPR(conn *github.Client) *PR {
	return &PR{conn: conn}
}

/*
GetPR retrieves a single pull request from a repository.

Uses owner, repository name, and PR number from the artifact's metadata.
Returns an error if the retrieval fails.
*/
func (pr *PR) GetPR(owner, name string, number int) (pullRequest *github.PullRequest, err error) {
	pullRequest, _, err = pr.conn.PullRequests.Get(
		context.Background(),
		owner,
		name,
		number,
	)
	if err != nil {
		return nil, errnie.Error(err)
	}

	return pullRequest, nil
}

/*
ListPRs retrieves all pull requests from a repository.

Uses owner and repository name from the artifact's metadata.
Returns an error if the retrieval fails.
*/
func (pr *PR) ListPRs(owner, name string) (prs []*github.PullRequest, err error) {
	pullRequests, _, err := pr.conn.PullRequests.List(
		context.Background(),
		owner,
		name,
		nil,
	)
	if err != nil {
		return nil, errnie.Error(err)
	}
	return pullRequests, nil
}

/*
CreatePR creates a new pull request in a repository.

Uses metadata from the artifact to set PR fields like title, head branch,
base branch, and body. Returns an error if the creation fails.
*/
func (pr *PR) CreatePR(owner, name, title, head, base, body string) (createdPR *github.PullRequest, err error) {
	newPR := &github.NewPullRequest{
		Title:               github.Ptr(title),
		Head:                github.Ptr(head),
		Base:                github.Ptr(base),
		Body:                github.Ptr(body),
		MaintainerCanModify: github.Ptr(true),
	}

	createdPR, _, err = pr.conn.PullRequests.Create(
		context.Background(),
		owner,
		name,
		newPR,
	)
	if err != nil {
		return nil, errnie.Error(err)
	}
	return createdPR, nil
}

/*
UpdatePR updates an existing pull request in a repository.

Uses metadata from the artifact to update PR fields like title, body, and state.
Returns an error if the update fails.
*/
func (pr *PR) UpdatePR(owner, name string, number int, title, body, state string) (updatedPR *github.PullRequest, err error) {
	update := &github.PullRequest{
		Title: github.Ptr(title),
		Body:  github.Ptr(body),
		State: github.Ptr(state),
	}

	updatedPR, _, err = pr.conn.PullRequests.Edit(
		context.Background(),
		owner,
		name,
		number,
		update,
	)
	if err != nil {
		return nil, errnie.Error(err)
	}
	return updatedPR, nil
}

/*
CreatePRComment creates a new comment on a pull request.

Uses metadata from the artifact to set the comment body.
Returns an error if the comment creation fails.
*/
func (pr *PR) CreatePRComment(owner, name string, number int, body string) (createdComment *github.IssueComment, err error) {
	comment := &github.IssueComment{
		Body: github.Ptr(body),
	}

	createdComment, _, err = pr.conn.Issues.CreateComment(
		context.Background(),
		owner,
		name,
		number,
		comment,
	)
	if err != nil {
		return nil, errnie.Error(err)
	}
	return createdComment, nil
}

/*
ListPRComments retrieves all comments from a pull request.

Uses owner, repository name, and PR number from the artifact's metadata.
Returns an error if the retrieval fails.
*/
func (pr *PR) ListPRComments(owner, name string, number int) (listedComments []*github.IssueComment, err error) {
	listedComments, _, err = pr.conn.Issues.ListComments(
		context.Background(),
		owner,
		name,
		number,
		nil,
	)
	if err != nil {
		return nil, errnie.Error(err)
	}
	return listedComments, nil
}

/*
CreateReviewComment creates a new review comment on a specific line of code in a pull request.

Uses metadata from the artifact to set the comment body, file path, and position.
Returns an error if the comment creation fails.
*/
func (pr *PR) CreateReviewComment(owner, name string, number int, body, path string, position int) (createdComment *github.PullRequestComment, err error) {
	comment := &github.PullRequestComment{
		Body:     github.Ptr(body),
		Path:     github.Ptr(path),
		Position: github.Ptr(position),
	}

	createdComment, _, err = pr.conn.PullRequests.CreateComment(
		context.Background(),
		owner,
		name,
		number,
		comment,
	)
	if err != nil {
		return nil, errnie.Error(err)
	}
	return createdComment, nil
}

/*
ListReviewComments retrieves all review comments from a pull request.

Uses owner, repository name, and PR number from the artifact's metadata.
Returns an error if the retrieval fails.
*/
func (pr *PR) ListReviewComments(owner, name string, number int) (listedComments []*github.PullRequestComment, err error) {
	listedComments, _, err = pr.conn.PullRequests.ListComments(
		context.Background(),
		owner,
		name,
		number,
		nil,
	)
	if err != nil {
		return nil, errnie.Error(err)
	}
	return listedComments, nil
}

/*
CreatePRReview creates a new review on a pull request.

Uses metadata from the artifact to set the review body, event type (APPROVE,
REQUEST_CHANGES, COMMENT), and optional line-specific comments.
Returns an error if the review creation fails.
*/
func (pr *PR) CreatePRReview(owner, name string, number int, body, event string, path string, position int, comment string) (createdReview *github.PullRequestReview, err error) {
	review := &github.PullRequestReviewRequest{
		Body:  github.Ptr(body),
		Event: github.Ptr(event), // APPROVE, REQUEST_CHANGES, COMMENT
		Comments: []*github.DraftReviewComment{
			{
				Path:     github.Ptr(path),
				Position: github.Ptr(position),
				Body:     github.Ptr(comment),
			},
		},
	}

	createdReview, _, err = pr.conn.PullRequests.CreateReview(
		context.Background(),
		owner,
		name,
		number,
		review,
	)
	if err != nil {
		return nil, errnie.Error(err)
	}
	return createdReview, nil
}

/*
ListPRReviews retrieves all reviews from a pull request.

Uses owner, repository name, and PR number from the artifact's metadata.
Returns an error if the retrieval fails.
*/
func (pr *PR) ListPRReviews(owner, name string, number int) (listedReviews []*github.PullRequestReview, err error) {
	listedReviews, _, err = pr.conn.PullRequests.ListReviews(
		context.Background(),
		owner,
		name,
		number,
		nil,
	)
	if err != nil {
		return nil, errnie.Error(err)
	}
	return listedReviews, nil
}

/*
SubmitReview submits a pending review on a pull request.

Uses metadata from the artifact to set the review body and event type
(APPROVE, REQUEST_CHANGES, COMMENT). Returns an error if the submission fails.
*/
func (pr *PR) SubmitReview(owner, name string, number int, review_id int, body, event string) (submittedReview *github.PullRequestReview, err error) {
	review := &github.PullRequestReviewRequest{
		Body:  github.Ptr(body),
		Event: github.Ptr(event), // APPROVE, REQUEST_CHANGES, COMMENT
	}

	submittedReview, _, err = pr.conn.PullRequests.SubmitReview(
		context.Background(),
		owner,
		name,
		number,
		int64(review_id),
		review,
	)
	if err != nil {
		return nil, errnie.Error(err)
	}
	return submittedReview, nil
}
