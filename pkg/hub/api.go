package hub

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"strings"
)

type repositoryPayload struct {
	ID       string           `json:"id"`
	SHA      string           `json:"sha"`
	Siblings []siblingPayload `json:"siblings"`
}

type siblingPayload struct {
	RFilename string      `json:"rfilename"`
	Size      int64       `json:"size"`
	LFS       *lfsPayload `json:"lfs"`
}

type lfsPayload struct {
	SHA256      string `json:"sha256"`
	Size        int64  `json:"size"`
	PointerSize int64  `json:"pointerSize"`
}

func (client *Client) Repository(
	ctx context.Context, repoType RepoType, repoID, revision, token string,
) (*Repository, error) {
	repoType, err := parseRepoType(string(repoType))

	if err != nil {
		return nil, err
	}

	revision = normalizeRevision(revision)
	apiPlural, err := repoType.apiPlural()

	if err != nil {
		return nil, err
	}

	endpoint := strings.TrimRight(client.config.Endpoint, "/")
	requestURL := fmt.Sprintf(
		"%s/api/%s/%s/revision/%s?blobs=true",
		endpoint,
		apiPlural,
		escapeRepoID(repoID),
		url.PathEscape(revision),
	)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL, nil)

	if err != nil {
		return nil, fmt.Errorf("hub: build repository request: %w", err)
	}

	authorize(req, token)

	resp, err := client.httpClient.Do(req)

	if err != nil {
		return nil, fmt.Errorf("hub: repository %s: %w", repoID, err)
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, statusError("hub: repository", resp)
	}

	var payload repositoryPayload

	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("hub: decode repository: %w", err)
	}

	return repositoryFromPayload(repoType, revision, repoID, payload)
}

func repositoryFromPayload(
	repoType RepoType,
	revision string,
	fallbackID string,
	payload repositoryPayload,
) (*Repository, error) {
	if payload.SHA == "" {
		return nil, fmt.Errorf("hub: repository %s returned no commit sha", fallbackID)
	}

	siblings := make([]Sibling, 0, len(payload.Siblings))

	for _, sibling := range payload.Siblings {
		entry := Sibling{
			Filename: sibling.RFilename,
			Size:     sibling.Size,
		}

		if sibling.LFS != nil {
			entry.LFS = &LFSMetadata{
				SHA256:      sibling.LFS.SHA256,
				Size:        sibling.LFS.Size,
				PointerSize: sibling.LFS.PointerSize,
			}

			if entry.Size == 0 {
				entry.Size = sibling.LFS.Size
			}
		}

		if entry.Filename != "" {
			siblings = append(siblings, entry)
		}
	}

	repoID := payload.ID

	if repoID == "" {
		repoID = fallbackID
	}

	return &Repository{
		ID:       repoID,
		RepoType: repoType,
		Revision: revision,
		Commit:   payload.SHA,
		Siblings: siblings,
	}, nil
}

func resolveURL(
	endpoint string, repoType RepoType, repoID, revision, filename string,
) (string, error) {
	prefix, err := repoType.resolvePrefix()

	if err != nil {
		return "", err
	}

	parts := []string{strings.TrimRight(endpoint, "/")}

	if prefix != "" {
		parts = append(parts, prefix)
	}

	parts = append(
		parts,
		escapeRepoID(repoID),
		"resolve",
		url.PathEscape(normalizeRevision(revision)),
		escapeFilePath(filename),
	)

	return strings.Join(parts, "/"), nil
}

func escapeRepoID(repoID string) string {
	parts := strings.Split(repoID, "/")

	for index := range parts {
		parts[index] = url.PathEscape(parts[index])
	}

	return strings.Join(parts, "/")
}

func escapeFilePath(filename string) string {
	parts := strings.Split(path.Clean(filename), "/")

	for index := range parts {
		parts[index] = url.PathEscape(parts[index])
	}

	return strings.Join(parts, "/")
}

func authorize(req *http.Request, token string) {
	if strings.TrimSpace(token) == "" {
		return
	}

	req.Header.Set("Authorization", "Bearer "+token)
}

func statusError(prefix string, resp *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	message := strings.TrimSpace(string(body))

	if message == "" {
		return fmt.Errorf("%s: HTTP %d", prefix, resp.StatusCode)
	}

	return fmt.Errorf("%s: HTTP %d: %s", prefix, resp.StatusCode, message)
}
