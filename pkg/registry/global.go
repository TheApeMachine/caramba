package registry

import (
	"context"
	"fmt"
	"sync"

	sdk "github.com/redis/go-redis/v9"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stores/redis"
)

var once sync.Once
var registryAmbient *Global

func init() {
	registryAmbient = NewGlobal(
		WithRedisConn(
			redis.NewConn(
				redis.WithAddr("redis:6379"),
			),
		),
	)
}

func GetAmbient() *Global {
	return registryAmbient
}

type Global struct {
	*redis.Conn
	store   map[string]any
	tools   map[string]any
	toolsMu sync.RWMutex
}

type GlobalOption func(*Global)

func NewGlobal(opts ...GlobalOption) *Global {
	once.Do(func() {
		registryAmbient = &Global{
			Conn: redis.NewConn(
				redis.WithAddr("redis:6379"),
			),
			store: make(map[string]any),
			tools: make(map[string]any),
		}

		for _, opt := range opts {
			opt(registryAmbient)
		}
	})

	return registryAmbient
}

func WithRedisConn(conn *redis.Conn) GlobalOption {
	return func(g *Global) {
		g.Conn = conn
	}
}

func (global *Global) Register(
	ctx context.Context, key string, value any,
) error {
	return global.Put(ctx, key, value)
}

func (global *Global) Unregister(
	ctx context.Context, key string) error {
	return global.Client.Del(ctx, key).Err()
}

func (global *Global) Get(
	ctx context.Context,
	key string,
	collector any,
) (err error) {
	errnie.Trace("Global.Get", "key", key)

	if global.Conn == nil {
		return errnie.New(errnie.WithErrorType(errnie.ResourceNotAvailableError))
	}

	if err = global.Client.HGetAll(ctx, key).Scan(collector); err != nil {
		return err
	}

	return nil
}

func (global *Global) Put(
	ctx context.Context, key string, value any,
) (err error) {
	errnie.Trace("Global.Put", "key", key, "value", value)

	if global.Conn == nil || global.Client == nil {
		return errnie.New(errnie.WithErrorType(errnie.ResourceNotAvailableError))
	}

	var cmdrs []sdk.Cmder

	if cmdrs, err = global.Client.Pipelined(ctx, func(rdb sdk.Pipeliner) error {
		switch v := value.(type) {
		case map[string]any:
			for k, val := range v {
				rdb.HSet(ctx, key, k, val)
			}
		case string:
			rdb.Set(ctx, key, v, 0)
		default:
			return fmt.Errorf("unsupported value type: %T", value)
		}
		return nil
	}); err != nil {
		return err
	}

	for _, cmdr := range cmdrs {
		if err = cmdr.Err(); err != nil {
			return err
		}
	}

	return nil
}

func (global *Global) Delete(
	ctx context.Context, key string,
) (err error) {
	var cmdrs []sdk.Cmder

	if cmdrs, err = global.Client.Pipelined(ctx, func(rdb sdk.Pipeliner) error {
		rdb.HDel(ctx, key)
		return nil
	}); err != nil {
		return err
	}

	for _, cmdr := range cmdrs {
		if err = cmdr.Err(); err != nil {
			return err
		}
	}

	return nil
}

func (g *Global) GetTool(name string) (any, error) {
	g.toolsMu.RLock()
	defer g.toolsMu.RUnlock()
	if tool, ok := g.tools[name]; ok {
		return tool, nil
	}
	return nil, fmt.Errorf("tool %s not found", name)
}

func (g *Global) RegisterTool(name string, tool any) {
	g.toolsMu.Lock()
	defer g.toolsMu.Unlock()
	g.tools[name] = tool
}
