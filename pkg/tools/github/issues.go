package github

import (
	"context"

	"github.com/google/go-github/v70/github"
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
GetIssue retrieves a single issue from a repository.

Uses owner, repository name, and issue number from the artifact's metadata.
Returns an error if the retrieval fails.
*/
func (issues *Issues) GetIssue(issueID int) (issue *github.Issue, err error) {
	issue, _, err = issues.conn.Issues.Get(
		context.Background(),
		"owner",
		"name",
		issueID,
	)
	if err != nil {
		return nil, errnie.Error(err)
	}
	return issue, nil
}

/*
ListIssues retrieves all issues from a repository.

Uses owner and repository name from the artifact's metadata.
Returns an error if the retrieval fails.
*/
func (issues *Issues) ListIssues() (issueList []*github.Issue, err error) {
	issueList, _, err = issues.conn.Issues.ListByRepo(
		context.Background(),
		"owner",
		"name",
		nil,
	)
	if err != nil {
		return nil, errnie.Error(err)
	}
	return issueList, nil
}

/*
CreateIssue creates a new issue in a repository.

Uses metadata from the artifact to set issue fields like title and body.
Returns an error if the creation fails.
*/
func (issues *Issues) CreateIssue(title, body string) (issue *github.Issue, err error) {
	issueRequest := &github.IssueRequest{
		Title:     github.Ptr(title),
		Body:      github.Ptr(body),
		Labels:    &[]string{},
		Assignees: &[]string{},
	}

	issue, _, err = issues.conn.Issues.Create(
		context.Background(),
		"owner",
		"name",
		issueRequest,
	)
	if err != nil {
		return nil, errnie.Error(err)
	}
	return issue, nil
}

/*
UpdateIssue updates an existing issue in a repository.

Uses metadata from the artifact to update issue fields like title, body, and state.
Returns an error if the update fails.
*/
func (issues *Issues) UpdateIssue(issueID int, title, body, state string) (issue *github.Issue, err error) {
	update := &github.IssueRequest{
		Title: github.Ptr(title),
		Body:  github.Ptr(body),
		State: github.Ptr(state),
	}

	issue, _, err = issues.conn.Issues.Edit(
		context.Background(),
		"owner",
		"name",
		issueID,
		update,
	)
	if err != nil {
		return nil, errnie.Error(err)
	}
	return issue, nil
}
