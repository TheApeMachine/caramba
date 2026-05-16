package weights

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/theapemachine/caramba/pkg/config"
	"github.com/theapemachine/caramba/pkg/hub"
)

const (
	defaultSafeTensorsFile  = "model.safetensors"
	defaultSafeTensorsIndex = "model.safetensors.index.json"
)

type Source struct {
	Source   string
	Cache    string
	Revision string
	RepoType string
}

type Store struct {
	mu      sync.Mutex
	tensors map[string]tensorRef
	cache   map[string][]float64
	derived map[string][]float64
}

type TensorInfo struct {
	Name  string
	DType string
	Shape []int
}

type tensorRef struct {
	file   *safeTensorsFile
	tensor safeTensor
}

type safeTensorsIndex struct {
	WeightMap map[string]string `json:"weight_map"`
}

func Resolve(ctx context.Context, source Source) (*Store, error) {
	source.Source = strings.TrimSpace(source.Source)

	if source.Source == "" {
		return nil, fmt.Errorf("weights: source is required")
	}

	if paths, err := source.localPaths(); err == nil {
		return Open(paths...)
	}

	return source.resolveHub(ctx)
}

func Open(paths ...string) (*Store, error) {
	if len(paths) == 0 {
		return nil, fmt.Errorf("weights: at least one safetensors file is required")
	}

	store := &Store{
		tensors: make(map[string]tensorRef),
		cache:   make(map[string][]float64),
		derived: make(map[string][]float64),
	}

	for _, path := range paths {
		file, err := openSafeTensors(path)

		if err != nil {
			return nil, err
		}

		for name, tensor := range file.tensors {
			if _, exists := store.tensors[name]; exists {
				return nil, fmt.Errorf("weights: duplicate tensor %q", name)
			}

			store.tensors[name] = tensorRef{
				file:   file,
				tensor: tensor,
			}
		}
	}

	return store, nil
}

func (store *Store) Names() []string {
	names := make([]string, 0, len(store.tensors))

	for name := range store.tensors {
		names = append(names, name)
	}

	sort.Strings(names)

	return names
}

func (store *Store) Info(name string) (TensorInfo, bool) {
	ref, ok := store.tensors[name]

	if !ok {
		return TensorInfo{}, false
	}

	return TensorInfo{
		Name:  name,
		DType: ref.tensor.DType,
		Shape: append([]int(nil), ref.tensor.Shape...),
	}, true
}

func (store *Store) Values(name string) ([]float64, error) {
	store.mu.Lock()

	if values, ok := store.cache[name]; ok {
		store.mu.Unlock()

		return values, nil
	}

	store.mu.Unlock()

	ref, ok := store.tensors[name]

	if !ok {
		return nil, fmt.Errorf("weights: tensor %q not found", name)
	}

	values, err := ref.file.values(ref.tensor)

	if err != nil {
		return nil, err
	}

	store.mu.Lock()
	store.cache[name] = values
	store.mu.Unlock()

	return values, nil
}

func (store *Store) Derived(
	key string,
	build func() ([]float64, error),
) ([]float64, error) {
	store.mu.Lock()

	if values, ok := store.derived[key]; ok {
		store.mu.Unlock()

		return values, nil
	}

	store.mu.Unlock()

	values, err := build()

	if err != nil {
		return nil, err
	}

	store.mu.Lock()
	store.derived[key] = values
	store.mu.Unlock()

	return values, nil
}

func (store *Store) Has(name string) bool {
	_, ok := store.tensors[name]

	return ok
}

func (source Source) localPaths() ([]string, error) {
	if filepath.IsAbs(source.Source) || strings.HasPrefix(source.Source, "./") {
		return localPaths(source.Source)
	}

	if _, err := os.Stat(source.Source); err == nil {
		return localPaths(source.Source)
	}

	return nil, os.ErrNotExist
}

func localPaths(source string) ([]string, error) {
	info, err := os.Stat(source)

	if err != nil {
		return nil, err
	}

	if !info.IsDir() {
		if strings.HasSuffix(source, ".json") {
			return pathsFromIndex(filepath.Dir(source), source)
		}

		return []string{source}, nil
	}

	monolith := filepath.Join(source, defaultSafeTensorsFile)

	if _, err := os.Stat(monolith); err == nil {
		return []string{monolith}, nil
	}

	indexPath := filepath.Join(source, defaultSafeTensorsIndex)

	if _, err := os.Stat(indexPath); err == nil {
		return pathsFromIndex(source, indexPath)
	}

	matches, err := filepath.Glob(filepath.Join(source, "*.safetensors"))

	if err != nil || len(matches) == 0 {
		return nil, fmt.Errorf("weights: no safetensors files found in %s", source)
	}

	sort.Strings(matches)

	return matches, nil
}

func pathsFromIndex(root string, indexPath string) ([]string, error) {
	data, err := os.ReadFile(indexPath)

	if err != nil {
		return nil, fmt.Errorf("weights: read index %s: %w", indexPath, err)
	}

	var index safeTensorsIndex

	if err := json.Unmarshal(data, &index); err != nil {
		return nil, fmt.Errorf("weights: parse index %s: %w", indexPath, err)
	}

	seen := make(map[string]bool, len(index.WeightMap))
	paths := make([]string, 0, len(index.WeightMap))

	for _, file := range index.WeightMap {
		path := filepath.Join(root, file)

		if seen[path] {
			continue
		}

		seen[path] = true
		paths = append(paths, path)
	}

	sort.Strings(paths)

	return paths, nil
}

func (source Source) resolveHub(ctx context.Context) (*Store, error) {
	monolith, err := source.download(ctx, defaultSafeTensorsFile)

	if err == nil {
		return Open(monolith)
	}

	indexPath, indexErr := source.download(ctx, defaultSafeTensorsIndex)

	if indexErr != nil {
		return nil, fmt.Errorf("weights: download %s: %w", defaultSafeTensorsFile, err)
	}

	paths, err := source.downloadIndexShards(ctx, indexPath)

	if err != nil {
		return nil, err
	}

	return Open(paths...)
}

func (source Source) downloadIndexShards(ctx context.Context, indexPath string) ([]string, error) {
	data, err := os.ReadFile(indexPath)

	if err != nil {
		return nil, fmt.Errorf("weights: read index %s: %w", indexPath, err)
	}

	var index safeTensorsIndex

	if err := json.Unmarshal(data, &index); err != nil {
		return nil, fmt.Errorf("weights: parse index %s: %w", indexPath, err)
	}

	seen := make(map[string]bool, len(index.WeightMap))
	paths := make([]string, 0, len(index.WeightMap))

	for _, filename := range index.WeightMap {
		if seen[filename] {
			continue
		}

		seen[filename] = true

		path, err := source.download(ctx, filename)

		if err != nil {
			return nil, err
		}

		paths = append(paths, path)
	}

	sort.Strings(paths)

	return paths, nil
}

func (source Source) download(ctx context.Context, filename string) (string, error) {
	location, err := hub.ParseLocator(source.Source)

	if err != nil {
		return "", err
	}

	if source.Revision != "" {
		location.Revision = source.Revision
	}

	if source.RepoType != "" {
		repoType, err := parseRepoType(source.RepoType)

		if err != nil {
			return "", err
		}

		location.RepoType = repoType
	}

	hubConfig := config.NewHubConfig()

	if source.Cache != "" {
		hubConfig.CacheDir = source.Cache
	}

	file, err := hub.NewClient(hubConfig).Download(ctx, hub.DownloadRequest{
		RepoID:   location.RepoID,
		RepoType: location.RepoType,
		Revision: location.Revision,
		Filename: filename,
	})

	if err != nil {
		return "", err
	}

	return file.Path, nil
}

func parseRepoType(value string) (hub.RepoType, error) {
	switch hub.RepoType(strings.TrimSpace(value)) {
	case "", hub.ModelRepo:
		return hub.ModelRepo, nil
	case hub.DatasetRepo:
		return hub.DatasetRepo, nil
	case hub.SpaceRepo:
		return hub.SpaceRepo, nil
	default:
		return "", fmt.Errorf("weights: unsupported repo_type %q", value)
	}
}
