package redis

import (
	"context"

	"github.com/redis/go-redis/v9"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type Conn struct {
	*redis.Client
}

type ConnOption func(*Conn)

func NewConn(opts ...ConnOption) *Conn {
	conn := &Conn{}

	for _, opt := range opts {
		opt(conn)
	}

	return conn
}

func (conn *Conn) Get(ctx context.Context, key string) (string, error) {
	return conn.Client.Get(ctx, key).Result()
}

func (conn *Conn) Set(ctx context.Context, key string, value string) error {
	return conn.Client.Set(ctx, key, value, 0).Err()
}

func WithAddr(addr string) ConnOption {
	return func(c *Conn) {
		var (
			rdb = redis.NewClient(&redis.Options{
				Addr: addr,
			})
			ctx = context.Background()
			err error
		)

		if err = rdb.Ping(ctx).Err(); err != nil {
			errnie.Error(errnie.WithError(err))
		}

		if err = rdb.FlushDB(ctx).Err(); err != nil {
			errnie.Error(errnie.WithError(err))
		}

		c.Client = rdb
	}
}
