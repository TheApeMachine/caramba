package hub

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"golang.org/x/sync/errgroup"
)

type remoteMetadata struct {
	ETag    string
	XetHash string
	Size    int64
}

/*
Download downloads one file and returns its immutable snapshot path.
*/
func (client *Client) Download(
	ctx context.Context, request DownloadRequest,
) (*File, error) {
	request = client.normalizeDownloadRequest(request)

	if err := validateDownloadRequest(request); err != nil {
		return nil, err
	}

	paths := newCachePaths(request.CacheDir, request.RepoType, request.RepoID)

	if client.config.Offline {
		return offlineFile(paths, request)
	}

	repository, err := client.Repository(
		ctx,
		request.RepoType,
		request.RepoID,
		request.Revision,
		request.Token,
	)

	if err != nil {
		return nil, err
	}

	return client.downloadWithRepository(ctx, request, repository, paths)
}

func (client *Client) downloadWithRepository(
	ctx context.Context,
	request DownloadRequest,
	repository *Repository,
	paths cachePaths,
) (*File, error) {
	if err := paths.ensure(); err != nil {
		return nil, err
	}

	if err := paths.writeRef(request.Revision, repository.Commit); err != nil {
		return nil, err
	}

	snapshotPath := paths.snapshotFile(repository.Commit, request.Filename)
	file := &File{
		RepoID:        request.RepoID,
		RepoType:      request.RepoType,
		Revision:      request.Revision,
		Commit:        repository.Commit,
		Filename:      request.Filename,
		Path:          snapshotPath,
		WouldDownload: true,
	}

	if !request.Force {
		if info, err := os.Stat(snapshotPath); err == nil {
			file.Size = info.Size()
			file.Cached = true
			file.WouldDownload = false

			return file, nil
		}
	}

	if request.DryRun {
		if sibling, ok := repository.Find(request.Filename); ok {
			file.Size = sibling.Size
		}

		return file, nil
	}

	tmpPath := filepath.Join(paths.tmp, sanitizeIdentity(request.Filename)+"."+strconv.FormatInt(time.Now().UnixNano(), 10))

	metadata, err := client.downloadTo(
		ctx,
		request,
		tmpPath,
	)

	if err != nil {
		_ = os.Remove(tmpPath)

		return nil, err
	}

	identity := metadata.XetHash

	if identity == "" {
		identity = metadata.ETag
	}

	blobPath, sha, err := installBlob(paths, tmpPath, identity)

	if err != nil {
		return nil, err
	}

	if err := linkSnapshot(blobPath, snapshotPath); err != nil {
		return nil, err
	}

	info, err := os.Stat(snapshotPath)

	if err != nil {
		return nil, fmt.Errorf("hub: stat snapshot: %w", err)
	}

	file.BlobPath = blobPath
	file.ETag = metadata.ETag
	file.XetHash = metadata.XetHash
	file.SHA256 = sha
	file.Size = info.Size()
	file.DownloadedAt = time.Now().UTC()

	if err := paths.writeMetadata(*file); err != nil {
		return nil, err
	}

	return file, nil
}

/*
Snapshot downloads all matching files for a repository revision.
*/
func (client *Client) Snapshot(
	ctx context.Context, request SnapshotRequest,
) (*Snapshot, error) {
	request = client.normalizeSnapshotRequest(request)

	if err := validateSnapshotRequest(request); err != nil {
		return nil, err
	}

	paths := newCachePaths(request.CacheDir, request.RepoType, request.RepoID)

	if client.config.Offline {
		return offlineSnapshot(paths, request)
	}

	repository, err := client.Repository(
		ctx,
		request.RepoType,
		request.RepoID,
		request.Revision,
		request.Token,
	)

	if err != nil {
		return nil, err
	}

	matches := repository.Matching(request.Include, request.Exclude)
	files := make([]File, len(matches))
	group, groupCtx := errgroup.WithContext(ctx)
	sem := make(chan struct{}, request.MaxWorkers)

	for index, sibling := range matches {
		index := index
		sibling := sibling

		group.Go(func() error {
			sem <- struct{}{}
			defer func() {
				<-sem
			}()

			file, err := client.downloadWithRepository(groupCtx, DownloadRequest{
				RepoID:   request.RepoID,
				RepoType: request.RepoType,
				Revision: request.Revision,
				Filename: sibling.Filename,
				CacheDir: request.CacheDir,
				Token:    request.Token,
				Force:    request.Force,
				DryRun:   request.DryRun,
			}, repository, paths)

			if err != nil {
				return err
			}

			files[index] = *file

			return nil
		})
	}

	if err := group.Wait(); err != nil {
		return nil, err
	}

	return &Snapshot{
		RepoID:   request.RepoID,
		RepoType: request.RepoType,
		Revision: request.Revision,
		Commit:   repository.Commit,
		Path:     paths.snapshotDir(repository.Commit),
		Files:    files,
	}, nil
}

func (client *Client) normalizeDownloadRequest(
	request DownloadRequest,
) DownloadRequest {
	if request.RepoType == "" {
		request.RepoType = ModelRepo
	}

	request.Revision = normalizeRevision(request.Revision)

	if request.CacheDir == "" {
		request.CacheDir = client.config.CacheDir
	}

	if request.Token == "" {
		request.Token = client.config.Token
	}

	return request
}

func (client *Client) normalizeSnapshotRequest(
	request SnapshotRequest,
) SnapshotRequest {
	if request.RepoType == "" {
		request.RepoType = ModelRepo
	}

	request.Revision = normalizeRevision(request.Revision)

	if request.CacheDir == "" {
		request.CacheDir = client.config.CacheDir
	}

	if request.Token == "" {
		request.Token = client.config.Token
	}

	if request.MaxWorkers <= 0 {
		request.MaxWorkers = client.config.MaxWorkers
	}

	if request.MaxWorkers <= 0 {
		request.MaxWorkers = 1
	}

	return request
}

func validateDownloadRequest(request DownloadRequest) error {
	if request.RepoID == "" {
		return fmt.Errorf("hub: repo id is required")
	}

	if request.Filename == "" {
		return fmt.Errorf("hub: filename is required")
	}

	if _, err := parseRepoType(string(request.RepoType)); err != nil {
		return err
	}

	return nil
}

func validateSnapshotRequest(request SnapshotRequest) error {
	if request.RepoID == "" {
		return fmt.Errorf("hub: repo id is required")
	}

	if _, err := parseRepoType(string(request.RepoType)); err != nil {
		return err
	}

	return nil
}

func offlineFile(paths cachePaths, request DownloadRequest) (*File, error) {
	commit, err := readOfflineCommit(paths, request.Revision)

	if err != nil {
		return nil, err
	}

	path := paths.snapshotFile(commit, request.Filename)
	info, err := os.Stat(path)

	if err != nil {
		return nil, fmt.Errorf("hub: offline file %s is not cached: %w", request.Filename, err)
	}

	return &File{
		RepoID:        request.RepoID,
		RepoType:      request.RepoType,
		Revision:      request.Revision,
		Commit:        commit,
		Filename:      request.Filename,
		Path:          path,
		Size:          info.Size(),
		Cached:        true,
		WouldDownload: false,
	}, nil
}

func offlineSnapshot(paths cachePaths, request SnapshotRequest) (*Snapshot, error) {
	commit, err := readOfflineCommit(paths, request.Revision)

	if err != nil {
		return nil, err
	}

	root := paths.snapshotDir(commit)
	files := make([]File, 0)

	err = filepath.WalkDir(root, func(filePath string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil || entry.IsDir() {
			return walkErr
		}

		filename, err := filepath.Rel(root, filePath)

		if err != nil {
			return err
		}

		filename = filepath.ToSlash(filename)

		if !matchesAny(request.Include, filename, true) {
			return nil
		}

		if matchesAny(request.Exclude, filename, false) {
			return nil
		}

		info, err := entry.Info()

		if err != nil {
			return err
		}

		files = append(files, File{
			RepoID:        request.RepoID,
			RepoType:      request.RepoType,
			Revision:      request.Revision,
			Commit:        commit,
			Filename:      filename,
			Path:          filePath,
			Size:          info.Size(),
			Cached:        true,
			WouldDownload: false,
		})

		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("hub: offline snapshot: %w", err)
	}

	return &Snapshot{
		RepoID:   request.RepoID,
		RepoType: request.RepoType,
		Revision: request.Revision,
		Commit:   commit,
		Path:     root,
		Files:    files,
	}, nil
}

func readOfflineCommit(paths cachePaths, revision string) (string, error) {
	data, err := os.ReadFile(paths.refFile(revision))

	if err != nil {
		return "", fmt.Errorf("hub: offline revision %s is not cached: %w", revision, err)
	}

	commit := strings.TrimSpace(string(data))

	if commit == "" {
		return "", fmt.Errorf("hub: offline revision %s has an empty ref", revision)
	}

	return commit, nil
}

func (client *Client) downloadTo(
	ctx context.Context, request DownloadRequest, tmpPath string,
) (remoteMetadata, error) {
	probe, err := client.probe(ctx, request)

	if err != nil {
		return remoteMetadata{}, err
	}

	if probe.XetHash != "" && client.config.Xet.Active {
		return client.downloadXet(ctx, request, probe, tmpPath)
	}

	return client.downloadHTTP(ctx, request, tmpPath)
}

func (client *Client) probe(
	ctx context.Context, request DownloadRequest,
) (remoteMetadata, error) {
	head, err := client.probeWithMethod(ctx, request, http.MethodHead)

	if err != nil {
		return remoteMetadata{}, err
	}

	if head.Size != statusMethodUnsupported {
		if head.XetHash != "" || head.Size == statusOKNoRedirect {
			return head, nil
		}
	}

	return client.probeWithMethod(ctx, request, http.MethodGet)
}

const (
	statusMethodUnsupported = -2
	statusOKNoRedirect      = -1
)

func (client *Client) probeWithMethod(
	ctx context.Context, request DownloadRequest, method string,
) (remoteMetadata, error) {
	requestURL, err := resolveURL(
		client.config.Endpoint,
		request.RepoType,
		request.RepoID,
		request.Revision,
		request.Filename,
	)

	if err != nil {
		return remoteMetadata{}, err
	}

	req, err := http.NewRequestWithContext(ctx, method, requestURL, nil)

	if err != nil {
		return remoteMetadata{}, fmt.Errorf("hub: build probe: %w", err)
	}

	authorize(req, request.Token)

	resp, err := client.probeClient.Do(req)

	if err != nil {
		return remoteMetadata{}, fmt.Errorf("hub: probe %s: %w", request.Filename, err)
	}

	defer resp.Body.Close()

	if resp.StatusCode == http.StatusMethodNotAllowed || resp.StatusCode == http.StatusNotImplemented {
		return remoteMetadata{Size: statusMethodUnsupported}, nil
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 400 {
		return remoteMetadata{}, statusError("hub: probe", resp)
	}

	size := resp.ContentLength

	if resp.StatusCode == http.StatusOK {
		size = statusOKNoRedirect
	}

	return remoteMetadata{
		ETag:    resp.Header.Get("ETag"),
		XetHash: resp.Header.Get("X-Xet-Hash"),
		Size:    size,
	}, nil
}

func (client *Client) downloadHTTP(
	ctx context.Context, request DownloadRequest, tmpPath string,
) (remoteMetadata, error) {
	requestURL, err := resolveURL(
		client.config.Endpoint,
		request.RepoType,
		request.RepoID,
		request.Revision,
		request.Filename,
	)

	if err != nil {
		return remoteMetadata{}, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL, nil)

	if err != nil {
		return remoteMetadata{}, fmt.Errorf("hub: build download: %w", err)
	}

	authorize(req, request.Token)

	resp, err := client.httpClient.Do(req)

	if err != nil {
		return remoteMetadata{}, fmt.Errorf("hub: download %s: %w", request.Filename, err)
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return remoteMetadata{}, statusError("hub: download", resp)
	}

	if err := os.MkdirAll(filepath.Dir(tmpPath), 0o755); err != nil {
		return remoteMetadata{}, fmt.Errorf("hub: mkdir temp: %w", err)
	}

	file, err := os.Create(tmpPath)

	if err != nil {
		return remoteMetadata{}, fmt.Errorf("hub: create temp: %w", err)
	}

	defer file.Close()

	if _, err := io.Copy(file, resp.Body); err != nil {
		return remoteMetadata{}, fmt.Errorf("hub: write temp: %w", err)
	}

	return remoteMetadata{
		ETag:    resp.Header.Get("ETag"),
		XetHash: resp.Header.Get("X-Xet-Hash"),
		Size:    resp.ContentLength,
	}, nil
}

func (repository *Repository) Find(filename string) (Sibling, bool) {
	for _, sibling := range repository.Siblings {
		if sibling.Filename == filename {
			return sibling, true
		}
	}

	return Sibling{}, false
}

func (repository *Repository) Matching(
	include []string, exclude []string,
) []Sibling {
	matches := make([]Sibling, 0, len(repository.Siblings))

	for _, sibling := range repository.Siblings {
		if !matchesAny(include, sibling.Filename, true) {
			continue
		}

		if matchesAny(exclude, sibling.Filename, false) {
			continue
		}

		matches = append(matches, sibling)
	}

	return matches
}

func matchesAny(patterns []string, filename string, emptyValue bool) bool {
	if len(patterns) == 0 {
		return emptyValue
	}

	for _, pattern := range patterns {
		if glob(pattern, filename) {
			return true
		}
	}

	return false
}
