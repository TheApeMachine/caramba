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

			switch artifact.Role() {
			case uint32(datura.ArtifactRoleListRepositories):
				return repo.GetRepositories(artifact)
			case uint32(datura.ArtifactRoleGetRepository):
				return repo.GetRepository(artifact)
			case uint32(datura.ArtifactRoleCreateRepository):
				return repo.CreateRepository(artifact)
			case uint32(datura.ArtifactRoleListBranches):
				return repo.ListBranches(artifact)
			case uint32(datura.ArtifactRoleGetContents):
				return repo.GetContents(artifact)
			case uint32(datura.ArtifactRoleListPullRequests):
				return pr.ListPRs(artifact)
			case uint32(datura.ArtifactRoleGetPullRequest):
				return pr.GetPR(artifact)
			case uint32(datura.ArtifactRoleCreatePullRequest):
				return pr.CreatePR(artifact)
			case uint32(datura.ArtifactRoleUpdatePullRequest):
				return pr.UpdatePR(artifact)
			case uint32(datura.ArtifactRoleCreatePRComment):
				return pr.CreatePRComment(artifact)
			case uint32(datura.ArtifactRoleListPRComments):
				return pr.ListPRComments(artifact)
			case uint32(datura.ArtifactRoleCreatePRReview):
				return pr.CreatePRReview(artifact)
			case uint32(datura.ArtifactRoleListPRReviews):
				return pr.ListPRReviews(artifact)
			case uint32(datura.ArtifactRoleCreateReviewComment):
				return pr.CreateReviewComment(artifact)
			case uint32(datura.ArtifactRoleListReviewComments):
				return pr.ListReviewComments(artifact)
			case uint32(datura.ArtifactRoleSubmitReview):
				return pr.SubmitReview(artifact)
			case uint32(datura.ArtifactRoleListIssues):
				return issues.ListIssues(artifact)
			case uint32(datura.ArtifactRoleGetIssue):
				return issues.GetIssue(artifact)
			case uint32(datura.ArtifactRoleCreateIssue):
				return issues.CreateIssue(artifact)
			case uint32(datura.ArtifactRoleUpdateIssue):
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
