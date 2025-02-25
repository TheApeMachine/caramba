package tools

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	"github.com/google/go-github/v66/github"
	"golang.org/x/oauth2"
)

// GitHubTool provides integration with the GitHub API
type GitHubTool struct {
	// client is the GitHub API client
	client *github.Client
	// token is the GitHub access token
	token string
	// defaultOwner is the default repository owner/org
	defaultOwner string
	// defaultRepo is the default repository name
	defaultRepo string
}

// NewGitHubTool creates a new GitHubTool
func NewGitHubTool(token string, defaultOwner, defaultRepo string) *GitHubTool {
	// Create OAuth2 token source
	ts := oauth2.StaticTokenSource(
		&oauth2.Token{AccessToken: token},
	)
	tc := oauth2.NewClient(context.Background(), ts)

	// Create GitHub client
	client := github.NewClient(tc)

	return &GitHubTool{
		client:       client,
		token:        token,
		defaultOwner: defaultOwner,
		defaultRepo:  defaultRepo,
	}
}

// Name returns the name of the tool
func (g *GitHubTool) Name() string {
	return "github"
}

// Description returns the description of the tool
func (g *GitHubTool) Description() string {
	return "Integrates with GitHub API for code repositories, pull requests, issues, and more"
}

// Execute executes the tool with the given arguments
func (g *GitHubTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	action, ok := args["action"].(string)
	if !ok {
		return nil, errors.New("action must be a string")
	}

	switch action {
	case "search_code":
		return g.searchCode(ctx, args)
	case "get_repo":
		return g.getRepository(ctx, args)
	case "list_issues":
		return g.listIssues(ctx, args)
	case "create_issue":
		return g.createIssue(ctx, args)
	case "get_pull_request":
		return g.getPullRequest(ctx, args)
	case "list_pull_requests":
		return g.listPullRequests(ctx, args)
	case "create_pull_request":
		return g.createPullRequest(ctx, args)
	case "review_pull_request":
		return g.reviewPullRequest(ctx, args)
	case "get_file_content":
		return g.getFileContent(ctx, args)
	case "create_comment":
		return g.createComment(ctx, args)
	default:
		return nil, fmt.Errorf("unknown action: %s", action)
	}
}

// Schema returns the JSON schema for the tool's arguments
func (g *GitHubTool) Schema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"action": map[string]interface{}{
				"type": "string",
				"enum": []string{
					"search_code",
					"get_repo",
					"list_issues",
					"create_issue",
					"get_pull_request",
					"list_pull_requests",
					"create_pull_request",
					"review_pull_request",
					"get_file_content",
					"create_comment",
				},
				"description": "Action to perform with the GitHub API",
			},
			"owner": map[string]interface{}{
				"type":        "string",
				"description": "Repository owner (user or organization)",
			},
			"repo": map[string]interface{}{
				"type":        "string",
				"description": "Repository name",
			},
			"query": map[string]interface{}{
				"type":        "string",
				"description": "Search query (for search operations)",
			},
			"path": map[string]interface{}{
				"type":        "string",
				"description": "File path (for file operations)",
			},
			"title": map[string]interface{}{
				"type":        "string",
				"description": "Title (for issues, PRs)",
			},
			"body": map[string]interface{}{
				"type":        "string",
				"description": "Body content (for issues, PRs, comments)",
			},
			"number": map[string]interface{}{
				"type":        "number",
				"description": "Issue or PR number",
			},
			"state": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"open", "closed", "all"},
				"description": "State filter (for issues, PRs)",
			},
			"base": map[string]interface{}{
				"type":        "string",
				"description": "Base branch (for PRs)",
			},
			"head": map[string]interface{}{
				"type":        "string",
				"description": "Head branch (for PRs)",
			},
			"event": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"APPROVE", "REQUEST_CHANGES", "COMMENT"},
				"description": "Review event type (for PR reviews)",
			},
		},
		"required": []string{"action"},
	}
}

// getOwnerRepo gets the owner and repo from args or uses defaults
func (g *GitHubTool) getOwnerRepo(args map[string]interface{}) (string, string, error) {
	owner, ownerOk := args["owner"].(string)
	repo, repoOk := args["repo"].(string)

	if !ownerOk {
		if g.defaultOwner == "" {
			return "", "", errors.New("owner must be specified")
		}
		owner = g.defaultOwner
	}

	if !repoOk {
		if g.defaultRepo == "" {
			return "", "", errors.New("repo must be specified")
		}
		repo = g.defaultRepo
	}

	return owner, repo, nil
}

// searchCode searches for code in repositories
func (g *GitHubTool) searchCode(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	query, ok := args["query"].(string)
	if !ok || query == "" {
		return nil, errors.New("query must be a non-empty string")
	}

	// Handle optional owner/repo restriction
	owner, ownerOk := args["owner"].(string)
	repo, repoOk := args["repo"].(string)
	if ownerOk && repoOk {
		query = fmt.Sprintf("%s repo:%s/%s", query, owner, repo)
	} else if ownerOk {
		query = fmt.Sprintf("%s user:%s", query, owner)
	}

	// Handle optional language restriction
	if language, ok := args["language"].(string); ok && language != "" {
		query = fmt.Sprintf("%s language:%s", query, language)
	}

	// Handle optional path restriction
	if path, ok := args["path"].(string); ok && path != "" {
		query = fmt.Sprintf("%s path:%s", query, path)
	}

	// Perform the search
	opts := &github.SearchOptions{
		ListOptions: github.ListOptions{
			PerPage: 30,
		},
	}

	result, resp, err := g.client.Search.Code(ctx, query, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to search code: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GitHub API error, status: %d", resp.StatusCode)
	}

	// Convert to a simplified format
	items := make([]map[string]interface{}, 0, len(result.CodeResults))
	for _, item := range result.CodeResults {
		items = append(items, map[string]interface{}{
			"repository": item.Repository.GetFullName(),
			"path":       item.GetPath(),
			"url":        item.GetHTMLURL(),
			"sha":        item.GetSHA(),
		})
	}

	return map[string]interface{}{
		"total_count": result.GetTotal(),
		"items":       items,
	}, nil
}

// getRepository gets details about a repository
func (g *GitHubTool) getRepository(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	owner, repo, err := g.getOwnerRepo(args)
	if err != nil {
		return nil, err
	}

	repository, resp, err := g.client.Repositories.Get(ctx, owner, repo)
	if err != nil {
		return nil, fmt.Errorf("failed to get repository: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GitHub API error, status: %d", resp.StatusCode)
	}

	return map[string]interface{}{
		"name":        repository.GetName(),
		"full_name":   repository.GetFullName(),
		"description": repository.GetDescription(),
		"url":         repository.GetHTMLURL(),
		"stars":       repository.GetStargazersCount(),
		"forks":       repository.GetForksCount(),
		"open_issues": repository.GetOpenIssuesCount(),
		"language":    repository.GetLanguage(),
		"private":     repository.GetPrivate(),
		"created_at":  repository.GetCreatedAt(),
		"updated_at":  repository.GetUpdatedAt(),
	}, nil
}

// listIssues lists issues in a repository
func (g *GitHubTool) listIssues(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	owner, repo, err := g.getOwnerRepo(args)
	if err != nil {
		return nil, err
	}

	// Parse state
	state := "open"
	if stateArg, ok := args["state"].(string); ok && stateArg != "" {
		state = stateArg
	}

	opts := &github.IssueListByRepoOptions{
		State: state,
		ListOptions: github.ListOptions{
			PerPage: 30,
		},
	}

	issues, resp, err := g.client.Issues.ListByRepo(ctx, owner, repo, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to list issues: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GitHub API error, status: %d", resp.StatusCode)
	}

	// Convert to a simplified format
	items := make([]map[string]interface{}, 0, len(issues))
	for _, issue := range issues {
		// Skip pull requests (they also appear as issues in the API)
		if issue.IsPullRequest() {
			continue
		}

		items = append(items, map[string]interface{}{
			"number":     issue.GetNumber(),
			"title":      issue.GetTitle(),
			"state":      issue.GetState(),
			"created_at": issue.GetCreatedAt(),
			"updated_at": issue.GetUpdatedAt(),
			"url":        issue.GetHTMLURL(),
			"author":     issue.GetUser().GetLogin(),
		})
	}

	return map[string]interface{}{
		"total_count": len(items),
		"items":       items,
	}, nil
}

// createIssue creates a new issue
func (g *GitHubTool) createIssue(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	owner, repo, err := g.getOwnerRepo(args)
	if err != nil {
		return nil, err
	}

	title, ok := args["title"].(string)
	if !ok || title == "" {
		return nil, errors.New("title must be a non-empty string")
	}

	body, _ := args["body"].(string)

	issueRequest := &github.IssueRequest{
		Title: github.String(title),
		Body:  github.String(body),
	}

	// Handle labels if provided
	if labelsArg, ok := args["labels"].([]interface{}); ok {
		labels := make([]string, 0, len(labelsArg))
		for _, label := range labelsArg {
			if labelStr, ok := label.(string); ok {
				labels = append(labels, labelStr)
			}
		}
		issueRequest.Labels = &labels
	}

	issue, resp, err := g.client.Issues.Create(ctx, owner, repo, issueRequest)
	if err != nil {
		return nil, fmt.Errorf("failed to create issue: %w", err)
	}

	if resp.StatusCode != http.StatusCreated {
		return nil, fmt.Errorf("GitHub API error, status: %d", resp.StatusCode)
	}

	return map[string]interface{}{
		"number":     issue.GetNumber(),
		"title":      issue.GetTitle(),
		"url":        issue.GetHTMLURL(),
		"created_at": issue.GetCreatedAt(),
	}, nil
}

// getPullRequest gets details about a specific pull request
func (g *GitHubTool) getPullRequest(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	owner, repo, err := g.getOwnerRepo(args)
	if err != nil {
		return nil, err
	}

	number, ok := args["number"].(float64)
	if !ok {
		return nil, errors.New("number must be provided")
	}

	pr, resp, err := g.client.PullRequests.Get(ctx, owner, repo, int(number))
	if err != nil {
		return nil, fmt.Errorf("failed to get pull request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GitHub API error, status: %d", resp.StatusCode)
	}

	return map[string]interface{}{
		"number":      pr.GetNumber(),
		"title":       pr.GetTitle(),
		"state":       pr.GetState(),
		"body":        pr.GetBody(),
		"url":         pr.GetHTMLURL(),
		"author":      pr.GetUser().GetLogin(),
		"created_at":  pr.GetCreatedAt(),
		"updated_at":  pr.GetUpdatedAt(),
		"base_branch": pr.GetBase().GetRef(),
		"head_branch": pr.GetHead().GetRef(),
		"mergeable":   pr.GetMergeable(),
		"merged":      pr.GetMerged(),
	}, nil
}

// listPullRequests lists pull requests in a repository
func (g *GitHubTool) listPullRequests(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	owner, repo, err := g.getOwnerRepo(args)
	if err != nil {
		return nil, err
	}

	// Parse state
	state := "open"
	if stateArg, ok := args["state"].(string); ok && stateArg != "" {
		state = stateArg
	}

	opts := &github.PullRequestListOptions{
		State: state,
		ListOptions: github.ListOptions{
			PerPage: 30,
		},
	}

	prs, resp, err := g.client.PullRequests.List(ctx, owner, repo, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to list pull requests: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GitHub API error, status: %d", resp.StatusCode)
	}

	// Convert to a simplified format
	items := make([]map[string]interface{}, 0, len(prs))
	for _, pr := range prs {
		items = append(items, map[string]interface{}{
			"number":      pr.GetNumber(),
			"title":       pr.GetTitle(),
			"state":       pr.GetState(),
			"created_at":  pr.GetCreatedAt(),
			"updated_at":  pr.GetUpdatedAt(),
			"url":         pr.GetHTMLURL(),
			"author":      pr.GetUser().GetLogin(),
			"base_branch": pr.GetBase().GetRef(),
			"head_branch": pr.GetHead().GetRef(),
		})
	}

	return map[string]interface{}{
		"total_count": len(items),
		"items":       items,
	}, nil
}

// createPullRequest creates a new pull request
func (g *GitHubTool) createPullRequest(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	owner, repo, err := g.getOwnerRepo(args)
	if err != nil {
		return nil, err
	}

	title, ok := args["title"].(string)
	if !ok || title == "" {
		return nil, errors.New("title must be a non-empty string")
	}

	head, ok := args["head"].(string)
	if !ok || head == "" {
		return nil, errors.New("head branch must be specified")
	}

	base, ok := args["base"].(string)
	if !ok || base == "" {
		return nil, errors.New("base branch must be specified")
	}

	body, _ := args["body"].(string)

	pr, resp, err := g.client.PullRequests.Create(ctx, owner, repo, &github.NewPullRequest{
		Title: github.String(title),
		Head:  github.String(head),
		Base:  github.String(base),
		Body:  github.String(body),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create pull request: %w", err)
	}

	if resp.StatusCode != http.StatusCreated {
		return nil, fmt.Errorf("GitHub API error, status: %d", resp.StatusCode)
	}

	return map[string]interface{}{
		"number":     pr.GetNumber(),
		"title":      pr.GetTitle(),
		"url":        pr.GetHTMLURL(),
		"created_at": pr.GetCreatedAt(),
	}, nil
}

// reviewPullRequest submits a review on a pull request
func (g *GitHubTool) reviewPullRequest(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	owner, repo, err := g.getOwnerRepo(args)
	if err != nil {
		return nil, err
	}

	number, ok := args["number"].(float64)
	if !ok {
		return nil, errors.New("number must be provided")
	}

	body, _ := args["body"].(string)

	event, ok := args["event"].(string)
	if !ok || event == "" {
		event = "COMMENT" // Default to comment
	}

	review, resp, err := g.client.PullRequests.CreateReview(ctx, owner, repo, int(number), &github.PullRequestReviewRequest{
		Body:  github.String(body),
		Event: github.String(event),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create review: %w", err)
	}

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return nil, fmt.Errorf("GitHub API error, status: %d", resp.StatusCode)
	}

	return map[string]interface{}{
		"id":         review.GetID(),
		"state":      review.GetState(),
		"submitted":  review.GetSubmittedAt(),
		"author":     review.GetUser().GetLogin(),
		"pr_number":  number,
		"repository": fmt.Sprintf("%s/%s", owner, repo),
	}, nil
}

// getFileContent gets the content of a file from a repository
func (g *GitHubTool) getFileContent(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	owner, repo, err := g.getOwnerRepo(args)
	if err != nil {
		return nil, err
	}

	path, ok := args["path"].(string)
	if !ok || path == "" {
		return nil, errors.New("path must be a non-empty string")
	}

	// Get optional ref (branch, tag, or commit SHA)
	ref, _ := args["ref"].(string)

	fileContent, _, resp, err := g.client.Repositories.GetContents(ctx, owner, repo, path, &github.RepositoryContentGetOptions{
		Ref: ref,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get file content: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GitHub API error, status: %d", resp.StatusCode)
	}

	if fileContent == nil {
		return nil, fmt.Errorf("path does not refer to a file")
	}

	// Decode content from base64
	content, err := fileContent.GetContent()
	if err != nil {
		return nil, fmt.Errorf("failed to decode content: %w", err)
	}

	return map[string]interface{}{
		"name":         fileContent.GetName(),
		"path":         fileContent.GetPath(),
		"sha":          fileContent.GetSHA(),
		"size":         fileContent.GetSize(),
		"content":      content,
		"download_url": fileContent.GetDownloadURL(),
		"html_url":     fileContent.GetHTMLURL(),
	}, nil
}

// createComment creates a comment on an issue or pull request
func (g *GitHubTool) createComment(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	owner, repo, err := g.getOwnerRepo(args)
	if err != nil {
		return nil, err
	}

	number, ok := args["number"].(float64)
	if !ok {
		return nil, errors.New("number must be provided")
	}

	body, ok := args["body"].(string)
	if !ok || body == "" {
		return nil, errors.New("body must be a non-empty string")
	}

	comment, resp, err := g.client.Issues.CreateComment(ctx, owner, repo, int(number), &github.IssueComment{
		Body: github.String(body),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create comment: %w", err)
	}

	if resp.StatusCode != http.StatusCreated {
		return nil, fmt.Errorf("GitHub API error, status: %d", resp.StatusCode)
	}

	return map[string]interface{}{
		"id":         comment.GetID(),
		"body":       comment.GetBody(),
		"created_at": comment.GetCreatedAt(),
		"updated_at": comment.GetUpdatedAt(),
		"author":     comment.GetUser().GetLogin(),
		"url":        comment.GetHTMLURL(),
	}, nil
}
