package daemon

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/errnie"
)

func Start(ctx context.Context) error {
	errnie.Info("Starting daemon")

	// Start hyperkit VM which runs containerd inside
	hv := NewHyperkit()
	if err := hv.Start(ctx); err != nil {
		return fmt.Errorf("failed to start hyperkit: %w", err)
	}

	return nil
}
