package runtime

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute"
)

const (
	DefaultModelManifest     = "model/llm/gpt2.yml"
	DefaultDiffusionManifest = "model/diffusion/flux-2-klein-4b.yml"
)

/*
Generator streams a response for a prompt.
*/
type Generator interface {
	Generate(ctx context.Context, prompt string, emit func(string) error) error
}

/*
SessionRunner owns an entire interactive terminal session.
*/
type SessionRunner interface {
	RunSession(ctx context.Context, input io.Reader, output io.Writer) error
}

/*
Session connects terminal streams to a manifest-driven runtime.
*/
type Session struct {
	ctx       context.Context
	input     io.Reader
	output    io.Writer
	generator Generator
}

func NewSession(
	ctx context.Context,
	input io.Reader,
	output io.Writer,
	generator Generator,
) *Session {
	return &Session{
		ctx:       ctx,
		input:     input,
		output:    output,
		generator: generator,
	}
}

func (session *Session) Run() error {
	runner, ok := session.generator.(SessionRunner)

	if !ok {
		return fmt.Errorf("runtime/session: generator does not implement SessionRunner")
	}

	return runner.RunSession(session.ctx, session.input, session.output)
}

/*
ModelConfig describes the model manifest and runtime manifest used by
the chat-style runtime program.
*/
type ModelConfig struct {
	Runtime           string
	RuntimeManifest   string
	Backend           string
	Model             string
	ModelFile         string
	Tokenizer         string
	TokenizerFile     string
	Manifest          string
	Cache             string
	Revision          string
	RepoType          string
	ModelCache        string
	ModelRevision     string
	ModelRepoType     string
	TokenizerCache    string
	TokenizerRevision string
	TokenizerRepoType string
	Seed              int64
}

func (config ModelConfig) ComputeBackend() (*compute.Backend, error) {
	backendName := strings.ToLower(strings.TrimSpace(config.Backend))

	if backendName == "" {
		backendName = "cpu"
	}

	backendType, err := backendType(backendName)

	if err != nil {
		return nil, fmt.Errorf("runtime/model: %w", err)
	}

	return compute.NewBackend(backendType)
}

/*
DiffusionConfig describes the diffusion model manifest and runtime
manifest used by an image runtime program.
*/
type DiffusionConfig struct {
	Manifest        string
	RuntimeManifest string
	Prompt          string
	Output          string
	Runtime         string
	Backend         string
	Model           Source
	Tokenizer       Source
	TextEncoder     Source
	Transformer     Source
	VAE             Source
	Generation      GenerationConfig
	Scheduler       SchedulerConfig
}

func (config DiffusionConfig) ComputeBackend() (*compute.Backend, error) {
	backendName := strings.ToLower(strings.TrimSpace(config.Backend))

	if backendName == "" {
		backendName = "auto"
	}

	backendType, err := backendType(backendName)

	if err != nil {
		return nil, fmt.Errorf("runtime/diffusion: %w", err)
	}

	return compute.NewBackend(backendType)
}

/*
Source describes a Hub or local artifact source.
*/
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

type Result struct {
	Output string
	Width  int
	Height int
}
