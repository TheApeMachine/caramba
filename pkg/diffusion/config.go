package diffusion

import (
	"fmt"
	"runtime"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute"
)

const DefaultManifest = "model/diffusion/flux-2-klein-4b.yml"

type Source struct {
	Source   string
	File     string
	Cache    string
	Revision string
	RepoType string
	Manifest string
}

type GenerationConfig struct {
	Height            int
	Width             int
	LatentChannels    int
	LatentDownsample  int
	MaxSequenceLength int
	Seed              int64
	Output            string
	PromptTemplate    string
	PadTokenID        int
}

type SchedulerConfig struct {
	Type              string
	Steps             int
	NumTrainTimesteps int
	BaseImageSeqLen   int
	MaxImageSeqLen    int
	BaseShift         float64
	MaxShift          float64
	Shift             float64
	UseDynamicShift   bool
	TimeShiftType     string
	Stochastic        bool
}

type Config struct {
	Manifest    string
	Prompt      string
	Output      string
	Runtime     string
	Backend     string
	Model       Source
	Tokenizer   Source
	TextEncoder Source
	Transformer Source
	VAE         Source
	Generation  GenerationConfig
	Scheduler   SchedulerConfig
}

func (config Config) ValidateRuntime() error {
	runtimeName := strings.ToLower(strings.TrimSpace(config.Runtime))

	if runtimeName == "diffusion" {
		return nil
	}

	return fmt.Errorf("diffusion: unsupported manifest runtime %q", config.Runtime)
}

func (config Config) ComputeBackend() (*compute.Backend, error) {
	backendName := strings.ToLower(strings.TrimSpace(config.Backend))

	if backendName == "" {
		backendName = "auto"
	}

	backendType, err := backendType(backendName)

	if err != nil {
		return nil, err
	}

	return compute.NewBackend(backendType)
}

func backendType(backendName string) (compute.BackendType, error) {
	switch backendName {
	case "auto":
		return automaticBackendType(), nil
	case "cpu", "host":
		return compute.CPU, nil
	case "metal":
		return compute.METAL, nil
	case "cuda":
		return compute.CUDA, nil
	case "xla":
		return compute.XLA, nil
	default:
		return compute.CPU, fmt.Errorf("diffusion: unsupported backend %q", backendName)
	}
}

func automaticBackendType() compute.BackendType {
	switch runtime.GOOS {
	case "darwin":
		return compute.METAL
	case "linux":
		return compute.CUDA
	default:
		return compute.CPU
	}
}
