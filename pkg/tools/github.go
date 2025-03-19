package tools

import (
	"io"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tools/github"
)

func init() {
	provider.RegisterTool("github")
}

type Github struct {
	buffer *stream.Buffer
	client *github.Client
	Schema *provider.Tool
}

func NewGithub() *Github {
	client := github.NewClient()

	return &Github{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("github.Client.buffer")

			if _, err = io.Copy(client, artifact); err != nil {
				return errnie.Error(err)
			}

			if _, err = io.Copy(artifact, client); err != nil {
				return errnie.Error(err)
			}

			return nil
		}),
		client: client,
		Schema: provider.NewTool(
			provider.WithFunction(
				"github",
				"A tool for interacting with GitHub.",
			),
			provider.WithProperty(
				"operation",
				"string",
				"The operation to perform.",
				[]any{
					"get_repositories",
					"get_repository",
					"create_repository",
					"list_branches",
					"get_contents",
					"list_pull_requests",
					"get_pull_request",
					"create_pull_request",
					"update_pull_request",
					"list_issues",
					"get_issue",
					"create_issue",
					"update_issue",
					"create_pr_comment",
					"list_pr_comments",
					"create_pr_review",
					"list_pr_reviews",
					"create_review_comment",
					"list_review_comments",
				},
			),
			provider.WithRequired("operation"),
		),
	}
}

func (github *Github) Read(p []byte) (n int, err error) {
	errnie.Debug("github.Read")
	return github.buffer.Read(p)
}

func (github *Github) Write(p []byte) (n int, err error) {
	errnie.Debug("github.Write")
	return github.buffer.Write(p)
}

func (github *Github) Close() error {
	errnie.Debug("github.Close")
	return github.buffer.Close()
}
