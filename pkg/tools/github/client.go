package github

import (
	"os"

	"github.com/google/go-github/v70/github"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

type Client struct {
	buffer *stream.Buffer
	conn   *github.Client
	repo   *Repository
	pr     *PR
	issues *Issues
}

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

func (client *Client) Read(p []byte) (n int, err error) {
	errnie.Debug("github.Client.Read")
	return client.buffer.Read(p)
}

func (client *Client) Write(p []byte) (n int, err error) {
	errnie.Debug("github.Client.Write")
	return client.buffer.Write(p)
}

func (client *Client) Close() error {
	errnie.Debug("github.Client.Close")
	return client.buffer.Close()
}
