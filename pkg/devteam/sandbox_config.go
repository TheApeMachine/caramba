package devteam

import (
	"time"

	"github.com/docker/docker/api/types/container"
)

const (
	sandboxDefaultExecTimeout = 10 * time.Minute
	sandboxDefaultMemoryBytes = 4 << 30
	sandboxDefaultNanoCPUs    = 2_000_000_000
	sandboxDefaultPidsLimit   = 512
	sandboxGitNetwork         = "bridge"
)

/*
SandboxConfig holds everything the sandbox needs to bootstrap the repository
inside the container.
*/
type SandboxConfig struct {
	AgentID       string
	Image         string
	GitHubToken   string
	GitHubOwner   string
	GitHubRepo    string
	FeatureBranch string
	ExecTimeout   time.Duration
	MemoryBytes   int64
	NanoCPUs      int64
	PidsLimit     int64
}

func (sandbox *Sandbox) hostConfig() *container.HostConfig {
	pidsLimit := sandbox.pidsLimit()

	return &container.HostConfig{
		AutoRemove:     false,
		CapDrop:        []string{"ALL"},
		NetworkMode:    "none",
		ReadonlyRootfs: true,
		Resources: container.Resources{
			Memory:    sandbox.memoryBytes(),
			NanoCPUs:  sandbox.nanoCPUs(),
			PidsLimit: &pidsLimit,
		},
		SecurityOpt: []string{"no-new-privileges:true"},
		Tmpfs: map[string]string{
			"/go":        "rw,nosuid,size=2g",
			"/root":      "rw,nosuid,size=64m",
			"/tmp":       "rw,noexec,nosuid,size=512m",
			"/workspace": "rw,nosuid,size=2g",
		},
	}
}

func (sandbox *Sandbox) execTimeout() time.Duration {
	if sandbox.cfg.ExecTimeout > 0 {
		return sandbox.cfg.ExecTimeout
	}

	return sandboxDefaultExecTimeout
}

func (sandbox *Sandbox) memoryBytes() int64 {
	if sandbox.cfg.MemoryBytes > 0 {
		return sandbox.cfg.MemoryBytes
	}

	return sandboxDefaultMemoryBytes
}

func (sandbox *Sandbox) nanoCPUs() int64 {
	if sandbox.cfg.NanoCPUs > 0 {
		return sandbox.cfg.NanoCPUs
	}

	return sandboxDefaultNanoCPUs
}

func (sandbox *Sandbox) pidsLimit() int64 {
	if sandbox.cfg.PidsLimit > 0 {
		return sandbox.cfg.PidsLimit
	}

	return sandboxDefaultPidsLimit
}
