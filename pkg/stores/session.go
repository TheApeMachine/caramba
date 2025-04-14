package stores

import (
	"encoding/json"

	"github.com/theapemachine/caramba/pkg/stores/types"
)

type Session struct {
	instance types.Store
	query    *types.Query
	encoder  *json.Encoder
	decoder  *json.Decoder
}

func NewSession(store types.Store, query *types.Query) *Session {
	return &Session{
		instance: store,
		query:    query,
		encoder:  json.NewEncoder(store),
		decoder:  json.NewDecoder(store),
	}
}

func (session *Session) Read(p []byte) (n int, err error) {
	return session.instance.Read(p)
}

func (session *Session) Write(p []byte) (n int, err error) {
	return session.instance.Write(p)
}

func (session *Session) Close() error {
	return session.instance.Close()
}
