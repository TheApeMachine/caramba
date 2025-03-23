package daemon

import (
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/containerd/containerd/namespaces"
	"github.com/containerd/containerd/v2/client"
	"github.com/mdlayher/vsock"
	hyperkit "github.com/moby/hyperkit/go"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type Hyperkit struct {
	vm          *hyperkit.HyperKit
	stateDir    string
	vpnKitSock  string
	kernelImage string
	initrd      string
	client      *client.Client
	shutdown    chan struct{}
}

func NewHyperkit() *Hyperkit {
	errnie.Info("Creating hyperkit instance")

	home, _ := os.UserHomeDir()
	stateDir := filepath.Join(home, ".local/share/caramba/hyperkit")
	vpnKitSock := filepath.Join(home, ".local/share/caramba/hyperkit/vpnkit.sock")
	// Create run directory if it doesn't exist
	runDir := filepath.Join(home, ".local/run/containerd")
	os.MkdirAll(runDir, 0755)

	return &Hyperkit{
		stateDir:    stateDir,
		vpnKitSock:  vpnKitSock,
		kernelImage: filepath.Join(stateDir, "vmlinuz"),
		initrd:      filepath.Join(stateDir, "initrd"),
	}
}

func (h *Hyperkit) buildLinuxKit() error {
	// Create linuxkit YAML config
	config := []byte(`
kernel:
  image: linuxkit/kernel:5.10.104
  cmdline: "console=ttyS0 vsock.no_cid_version_check=1"
init:
  - linuxkit/init:v0.8
  - linuxkit/runc:v0.8
  - linuxkit/containerd:v0.8
  - linuxkit/getty:v0.8
onboot:
  - name: sysctl
    image: linuxkit/sysctl:v0.8
  - name: dhcpcd
    image: linuxkit/dhcpcd:v0.8
    command: ["/sbin/dhcpcd", "--nobackground", "-f", "/dhcpcd.conf", "-1"]
services:
  - name: caramba-init
    image: debian:bookworm-slim
    command: ["/usr/local/bin/init.sh"]
    binds:
      - /var/run:/var/run
      - /run:/run
    capabilities:
      - all
    rootfsPropagation: shared
files:
  - path: /usr/local/bin/init.sh
    contents: |
      #!/bin/sh
      set -ex
      
      # Start containerd with our config
      mkdir -p /etc/containerd
      cat > /etc/containerd/config.toml << EOF
      version = 3
      root = "/var/lib/containerd"
      state = "/run/containerd"
      
      [grpc]
        address = "vsock://3:2376"
      
      [plugins]
        [plugins."io.containerd.runtime.v2.task"]
          platforms = ["linux/amd64", "linux/arm64"]
        [plugins."io.containerd.runtime.v1.linux"]
          shim = "containerd-shim"
          runtime = "runc"
      EOF
      
      # Start containerd
      containerd --config /etc/containerd/config.toml
`)

	// Write config
	if err := os.WriteFile(filepath.Join(h.stateDir, "linuxkit.yml"), config, 0644); err != nil {
		return fmt.Errorf("failed to write linuxkit config: %w", err)
	}

	// Build image using linuxkit
	cmd := exec.Command("linuxkit", "build", "-format", "kernel+initrd", filepath.Join(h.stateDir, "linuxkit.yml"))
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("failed to build linuxkit image: %w\noutput: %s", err, out)
	}

	return nil
}

func (h *Hyperkit) connectContainerd(ctx context.Context) error {
	// Wait for VM to be ready
	time.Sleep(5 * time.Second)

	// Test initial connection to ensure VM is ready
	testConn, err := vsock.Dial(3, 2376, &vsock.Config{})
	if err != nil {
		return fmt.Errorf("failed to connect to containerd via vsock: %w", err)
	}
	testConn.Close()

	// Create unix domain socket for containerd client
	home, _ := os.UserHomeDir()
	sockPath := filepath.Join(home, ".local/run/containerd/containerd.sock")
	if err := os.MkdirAll(filepath.Dir(sockPath), 0755); err != nil {
		return fmt.Errorf("failed to create socket directory: %w", err)
	}
	os.Remove(sockPath)
	l, err := net.Listen("unix", sockPath)
	if err != nil {
		return fmt.Errorf("failed to create unix socket: %w", err)
	}

	// Forward connections between unix socket and vsock
	go func() {
		for {
			c, err := l.Accept()
			if err != nil {
				errnie.Error(fmt.Errorf("failed to accept connection: %v", err))
				continue
			}
			go func(c net.Conn) {
				// Create new vsock connection for each client
				conn, err := vsock.Dial(3, 2376, &vsock.Config{})
				if err != nil {
					errnie.Error(fmt.Errorf("failed to connect to containerd via vsock: %v", err))
					c.Close()
					return
				}

				errCh := make(chan error, 2)
				go func() {
					_, err := io.Copy(c, conn)
					errCh <- err
					c.Close()
					conn.Close()
				}()
				go func() {
					_, err := io.Copy(conn, c)
					errCh <- err
					c.Close()
					conn.Close()
				}()

				<-errCh
			}(c)
		}
	}()

	// Connect containerd client to our forwarded socket
	for i := 0; i < 5; i++ {
		h.client, err = client.New(sockPath)
		if err == nil {
			break
		}
		errnie.Warn(fmt.Errorf("failed to connect to containerd, retrying in 1s: %v", err))
		time.Sleep(time.Second)
	}

	if h.client == nil {
		return fmt.Errorf("failed to connect to containerd after retries")
	}

	return nil
}

func (h *Hyperkit) Stop(ctx context.Context) error {
	errnie.Info("Stopping hyperkit")

	// Signal shutdown
	close(h.shutdown)

	// Close containerd client
	if h.client != nil {
		if err := h.client.Close(); err != nil {
			errnie.Warn(fmt.Errorf("error closing containerd client: %v", err))
		}
	}

	// Stop VM gracefully
	if h.vm != nil && h.vm.Pid > 0 {
		proc, err := os.FindProcess(h.vm.Pid)
		if err != nil {
			return fmt.Errorf("failed to find VM process: %w", err)
		}

		// Send SIGTERM first
		if err := proc.Signal(syscall.SIGTERM); err != nil {
			errnie.Warn(fmt.Errorf("failed to send SIGTERM to VM: %v", err))
		}

		// Wait up to 30 seconds for graceful shutdown
		done := make(chan struct{})
		go func() {
			proc.Wait()
			close(done)
		}()

		select {
		case <-done:
			errnie.Info("VM stopped gracefully")
		case <-time.After(30 * time.Second):
			errnie.Warn(fmt.Errorf("VM did not stop gracefully, forcing shutdown"))
			if err := proc.Kill(); err != nil {
				errnie.Error(fmt.Errorf("failed to kill VM: %v", err))
			}
		}
	}

	return nil
}

func (h *Hyperkit) PullImage(ctx context.Context, ref string) error {
	errnie.Info(fmt.Sprintf("Pulling image: %s", ref))

	ctx = namespaces.WithNamespace(ctx, "caramba")

	// Pull the image
	img, err := h.client.Pull(ctx, ref, client.WithPullUnpack)
	if err != nil {
		return fmt.Errorf("failed to pull image: %w", err)
	}

	errnie.Info(fmt.Sprintf("Successfully pulled image: %s", img.Name()))
	return nil
}

func (h *Hyperkit) ListImages(ctx context.Context) ([]client.Image, error) {
	ctx = namespaces.WithNamespace(ctx, "caramba")
	return h.client.ListImages(ctx)
}

func (h *Hyperkit) RemoveImage(ctx context.Context, ref string) error {
	ctx = namespaces.WithNamespace(ctx, "caramba")
	return h.client.ImageService().Delete(ctx, ref)
}

func (h *Hyperkit) Start(ctx context.Context) error {
	errnie.Info("Starting hyperkit")

	h.shutdown = make(chan struct{})

	var err error

	// Create state directory
	if err := os.MkdirAll(h.stateDir, 0755); err != nil {
		return fmt.Errorf("failed to create state directory: %w", err)
	}

	// Build Linux image if needed
	if _, err := os.Stat(h.kernelImage); os.IsNotExist(err) {
		if err := h.buildLinuxKit(); err != nil {
			return err
		}
	}

	// Initialize hyperkit
	h.vm, err = hyperkit.New("", h.vpnKitSock, h.stateDir)
	if err != nil {
		return fmt.Errorf("failed to create hyperkit instance: %w", err)
	}

	// Configure VM
	h.vm.CPUs = 2
	h.vm.Memory = 2048 // 2GB
	h.vm.VSock = true
	h.vm.VSockPorts = []int{2376} // Docker daemon port
	h.vm.VMNet = true
	h.vm.Kernel = h.kernelImage
	h.vm.Initrd = h.initrd

	// Start VM in background
	errCh, err := h.vm.Start("console=ttyS0")
	if err != nil {
		return fmt.Errorf("failed to start VM: %w", err)
	}

	// Monitor VM in background
	go func() {
		if err := <-errCh; err != nil {
			errnie.Error(fmt.Errorf("hyperkit exited with error: %v", err))
		}
	}()

	// Connect to containerd in VM
	if err := h.connectContainerd(ctx); err != nil {
		return err
	}

	// Handle OS signals for graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		select {
		case <-sigCh:
			errnie.Info("Received shutdown signal")
			h.Stop(ctx)
		case <-h.shutdown:
			// Normal shutdown, do nothing
		}
	}()

	return nil
}

// WaitForShutdown blocks until the VM is shut down
func (h *Hyperkit) WaitForShutdown() {
	<-h.shutdown
}
