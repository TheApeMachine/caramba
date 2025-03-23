package daemon

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/containerd/containerd/v2/client"
	"github.com/containerd/containerd/v2/cmd/containerd/command"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type Containerd struct {
	containerdRoot  string
	containerdState string
	containerdSock  string
}

func NewContainerd() *Containerd {
	errnie.Info("Creating containerd instance")

	home, _ := os.UserHomeDir()
	return &Containerd{
		containerdRoot:  filepath.Join(home, ".local/share/containerd"),
		containerdState: filepath.Join(home, ".local/run/containerd"),
		containerdSock:  filepath.Join(home, ".local/run/containerd/containerd.sock"),
	}
}

func (c *Containerd) Start(ctx context.Context) error {
	errnie.Info("Starting containerd")

	// Get current user info
	uid := os.Getuid()
	gid := os.Getgid()

	// Ensure directories exist with proper permissions
	dirs := []string{
		c.containerdRoot,
		c.containerdState,
		filepath.Join(c.containerdRoot, "snapshots"),
		filepath.Join(c.containerdRoot, "snapshots", "native"),
		filepath.Join(c.containerdRoot, "plugins"),
		filepath.Dir(c.containerdSock),
	}

	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("failed to create directory %s: %w", dir, err)
		}
		if err := os.Chown(dir, uid, gid); err != nil {
			return fmt.Errorf("failed to chown directory %s: %w", dir, err)
		}
	}

	// Set environment variables
	os.Setenv("CONTAINERD_ROOT", c.containerdRoot)
	os.Setenv("CONTAINERD_STATE", c.containerdState)
	os.Setenv("CONTAINERD_SNAPSHOTTER", "native")
	os.Setenv("XDG_RUNTIME_DIR", filepath.Dir(c.containerdState))

	// Create config file
	configPath := filepath.Join(c.containerdState, "config.toml")
	configContent := fmt.Sprintf(`
version = 3
root = %q
state = %q

disabled_plugins = [
    "io.containerd.snapshotter.v1.blockfile",
    "io.containerd.tracing.processor.v1.otlp",
    "io.containerd.internal.v1.tracing",
    "io.containerd.grpc.v1.cri"
]

[grpc]
  address = %q
  uid = %d
  gid = %d

[ttrpc]
  address = %q
  uid = %d
  gid = %d

[plugins]
  [plugins."io.containerd.runtime.v2.task"]
    platforms = ["linux/amd64", "linux/arm64"]
    shim_debug = true
    
  [plugins."io.containerd.runtime.v1.linux"]
    shim = "containerd-shim"
    runtime = "runc"
    runtime_root = %q
    no_shim = false
    shim_debug = true
    
  [plugins."io.containerd.snapshotter.v1.native"]
    root_path = %q

[plugins."io.containerd.snapshotter.v1"]
  default = "native"

[debug]
  level = "debug"

[proxy_plugins]
  [proxy_plugins.hyperkit]
    type = "snapshot"
    address = "unix://%s"
`, c.containerdRoot, c.containerdState, c.containerdSock, uid, gid, c.containerdSock+".ttrpc", uid, gid, c.containerdRoot+"/runc", filepath.Join(c.containerdRoot, "snapshots"), filepath.Join(c.containerdState, "hyperkit.sock"))

	if err := os.WriteFile(configPath, []byte(configContent), 0644); err != nil {
		return fmt.Errorf("failed to write config: %w", err)
	}
	if err := os.Chown(configPath, uid, gid); err != nil {
		return fmt.Errorf("failed to chown config: %w", err)
	}

	// Remove any existing socket files
	os.Remove(c.containerdSock)
	os.Remove(c.containerdSock + ".ttrpc")

	// Start containerd in background
	go func() {
		if err := command.App().RunContext(
			ctx,
			[]string{"containerd", "--config", configPath},
		); err != nil {
			fmt.Printf("containerd exited with error: %v\n", err)
		}
	}()

	return c.WaitForStart(ctx)
}

func (c *Containerd) WaitForStart(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	var isReady bool
	for !isReady {
		select {
		case <-ctx.Done():
			return fmt.Errorf("timeout waiting for containerd to start")
		default:
			conn, err := client.New(
				c.containerdSock,
				client.WithDefaultNamespace("caramba"),
			)

			if err != nil {
				time.Sleep(1 * time.Second)
				continue
			}

			conn.Close()
			isReady = true
		}
	}

	return nil
}
