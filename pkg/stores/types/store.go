package types

import "io"

type Store interface {
	Peek(*Query) (io.Reader, error)
	Poke(*Query) error
}

type Query struct {
	Collection string
	Filters    map[string]any
	Limit      int
	Offset     int
	Payload    io.ReadWriter
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

func WithPayload(payload io.ReadWriter) QueryOption {
	return func(query *Query) {
		query.Payload = payload
	}
}
