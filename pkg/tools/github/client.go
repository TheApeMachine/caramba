package github

import (
	"os"

	"github.com/google/go-github/v70/github"
)

/*
Client provides a high-level interface to GitHub services.
It manages connections and operations for repositories, pull requests,
and issues through a unified streaming interface.
*/
type Client struct {
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
		conn:   client,
		pr:     pr,
		repo:   repo,
		issues: issues,
	}
}

func (c *Client) Do(
	buffer chan []byte,
	fn ...func(artifact []byte) []byte,
) chan []byte {
	out := make(chan []byte)

	go func() {
		defer close(out)
	}()

	return out
}
