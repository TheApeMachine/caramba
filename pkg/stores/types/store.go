package types

import (
	"io"
)

type Store interface {
	io.ReadWriteCloser
}

type Query struct {
	Collection string
	Filters    map[string]any
	Limit      int
	Offset     int
	Payload    []byte
}

type QueryOption func(*Query)

func NewQuery(options ...QueryOption) *Query {
	query := &Query{
		Filters: make(map[string]any),
	}

	for _, option := range options {
		option(query)
	}

	return query
}

func (query *Query) HasFilters() bool {
	return len(query.Filters) > 0
}

func (query *Query) HasPayload() bool {
	return query.Payload != nil
}

func WithCollection(collection string) QueryOption {
	return func(query *Query) {
		query.Collection = collection
	}
}

func WithFilters(filters map[string]any) QueryOption {
	return func(query *Query) {
		query.Filters = filters
	}
}

func WithFilter(key string, value any) QueryOption {
	return func(query *Query) {
		query.Filters[key] = value
	}
}
