package weights

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/hf/hub"
	"gopkg.in/yaml.v3"
)

/*
programIncludes is a partial view of a manifest program manifest YAML — we
only care about the include block. Everything else is consumed by the
manifesto compiler.
*/
type programIncludes struct {
	Include map[string]string `yaml:"include"`
}

/*
HFReference identifies one Hugging Face Hub repository extracted from a
program manifest's include block.
*/
type HFReference struct {
	RepoID    string
	Component string
}

/*
ExtractHFReferences scans the program YAML for `hf://...` include entries
and returns each as a HFReference. References that don't parse cleanly are
skipped silently — the manifesto compiler will surface its own error for
malformed includes when it runs the same YAML through ParseHFReference.
*/
func ExtractHFReferences(programYAML []byte) ([]HFReference, error) {
	var parsed programIncludes

	if err := yaml.Unmarshal(programYAML, &parsed); err != nil {
		return nil, fmt.Errorf("weights loader: parse program YAML: %w", err)
	}

	refs := make([]HFReference, 0, len(parsed.Include))

	for _, source := range parsed.Include {
		ref, ok := parseHFReference(source)

		if !ok {
			continue
		}

		refs = append(refs, ref)
	}

	return refs, nil
}

/*
parseHFReference splits an `hf://repo[#component]` string into its parts.
*/
func parseHFReference(source string) (HFReference, bool) {
	trimmed := strings.TrimSpace(source)

	if !strings.HasPrefix(trimmed, "hf://") {
		return HFReference{}, false
	}

	body := strings.TrimPrefix(trimmed, "hf://")

	if body == "" {
		return HFReference{}, false
	}

	if hashIndex := strings.Index(body, "#"); hashIndex >= 0 {
		return HFReference{
			RepoID:    body[:hashIndex],
			Component: body[hashIndex+1:],
		}, true
	}

	return HFReference{RepoID: body}, true
}

/*
DownloadSafetensors materializes every *.safetensors sibling of the given
HF repository into the cache directory and returns the resolved snapshot
paths. The returned slice is empty (with a non-nil error) when the
repository has no safetensors archives.
*/
func DownloadSafetensors(
	ctx context.Context,
	client *hub.Client,
	repoID, revision, cacheDir, token string,
) ([]string, error) {
	if client == nil {
		return nil, fmt.Errorf("weights loader: hub client is required")
	}

	repository, err := client.Repository(ctx, hub.ModelRepo, repoID, revision, token)

	if err != nil {
		return nil, fmt.Errorf("weights loader: repository %q: %w", repoID, err)
	}

	paths := make([]string, 0)

	for _, sibling := range repository.Siblings {
		if !strings.HasSuffix(sibling.Filename, ".safetensors") {
			continue
		}

		file, err := client.Download(ctx, hub.DownloadRequest{
			RepoID:   repoID,
			RepoType: hub.ModelRepo,
			Revision: revision,
			Filename: sibling.Filename,
			CacheDir: cacheDir,
			Token:    token,
		})

		if err != nil {
			return nil, fmt.Errorf(
				"weights loader: download %q from %q: %w",
				sibling.Filename, repoID, err,
			)
		}

		paths = append(paths, file.Path)
	}

	if len(paths) == 0 {
		return nil, fmt.Errorf("weights loader: no safetensors files in %q", repoID)
	}

	return paths, nil
}
