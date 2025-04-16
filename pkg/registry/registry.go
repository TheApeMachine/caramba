package registry

import "context"

type Registry interface {
	Get(ctx context.Context, key string, collector any) error
	Register(ctx context.Context, key string, value any) error
	Unregister(ctx context.Context, key string) error
}
