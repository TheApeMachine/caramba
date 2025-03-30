package github

import (
	"os"

	"github.com/google/go-github/v70/github"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

/*
Client provides a high-level interface to GitHub services.
It manages connections and operations for repositories, pull requests,
and issues through a unified streaming interface.
*/
type Client struct {
	buffer *stream.Buffer
	conn   *github.Client
	repo   *Repository
	pr     *PR
	issues *Issues
}

/*
NewClient creates a new GitHub client using environment variables for authentication.

It initializes connections to repository, pull request, and issues services using
a GitHub personal access token from the GITHUB_TOKEN environment variable.
*/
func NewClient() *Client {
	client := github.NewClient(nil).WithAuthToken(os.Getenv("GITHUB_TOKEN"))
	repo := NewRepository(client)
	pr := NewPR(client)
	issues := NewIssues(client)

	return &Client{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("github.Client.buffer")

			operation := datura.GetMetaValue[string](artifact, "operation")

			switch operation {
			case "list_repositories":
				return repo.GetRepositories(artifact)
			case "get_repository":
				return repo.GetRepository(artifact)
			case "create_repository":
				return repo.CreateRepository(artifact)
			case "list_branches":
				return repo.ListBranches(artifact)
			case "get_contents":
				return repo.GetContents(artifact)
			case "list_pull_requests":
				return pr.ListPRs(artifact)
			case "get_pull_request":
				return pr.GetPR(artifact)
			case "create_pull_request":
				return pr.CreatePR(artifact)
			case "update_pull_request":
				return pr.UpdatePR(artifact)
			case "create_pr_comment":
				return pr.CreatePRComment(artifact)
			case "list_pr_comments":
				return pr.ListPRComments(artifact)
			case "create_pr_review":
				return pr.CreatePRReview(artifact)
			case "list_pr_reviews":
				return pr.ListPRReviews(artifact)
			case "create_review_comment":
				return pr.CreateReviewComment(artifact)
			case "list_review_comments":
				return pr.ListReviewComments(artifact)
			case "submit_review":
				return pr.SubmitReview(artifact)
			case "list_issues":
				return issues.ListIssues(artifact)
			case "get_issue":
				return issues.GetIssue(artifact)
			case "create_issue":
				return issues.CreateIssue(artifact)
			case "update_issue":
				return issues.UpdateIssue(artifact)
			}

			return nil
		}),
		conn:   client,
		pr:     pr,
		repo:   repo,
		issues: issues,
	}
}

/*
Read implements the io.Reader interface.

It reads processed data from the internal buffer after GitHub operations
have been completed.
*/
func (client *Client) Read(p []byte) (n int, err error) {
	errnie.Debug("github.Client.Read")
	return client.buffer.Read(p)
}

/*
Write implements the io.Writer interface.

It writes operation requests to the internal buffer for processing by
the appropriate GitHub service (repositories, pull requests, or issues).
*/
func (client *Client) Write(p []byte) (n int, err error) {
	errnie.Debug("github.Client.Write")
	return client.buffer.Write(p)
}

/*
Close implements the io.Closer interface.

It cleans up resources by closing the internal buffer.
*/
func (client *Client) Close() error {
	errnie.Debug("github.Client.Close")
	return client.buffer.Close()
}
