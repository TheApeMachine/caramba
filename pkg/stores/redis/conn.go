package redis

import (
	"context"

	"github.com/redis/go-redis/v9"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type Conn struct {
	*redis.Client
	err error
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
	if conn.err != nil {
		return "", conn.err
	}

	return conn.Client.Get(ctx, key).Result()
}

func (conn *Conn) Set(ctx context.Context, key string, value string) error {
	if conn.err != nil {
		return conn.err
	}

	return conn.Client.Set(ctx, key, value, 0).Err()
}

func WithAddr(addr string) ConnOption {
	return func(c *Conn) {
		var (
			rdb = redis.NewClient(&redis.Options{
				Addr: addr,
			})
			ctx = context.Background()
		)

		if c.err = rdb.Ping(ctx).Err(); c.err != nil {
			c.err = errnie.New(errnie.WithError(c.err))
		}

		if c.err = rdb.FlushDB(ctx).Err(); c.err != nil {
			c.err = errnie.New(errnie.WithError(c.err))
		}

		c.Client = rdb
	}
}
