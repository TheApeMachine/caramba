package inmemory

import (
	"io"
	"sync"

	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stores/types"
	"github.com/theapemachine/caramba/pkg/task"
)

var (
	once     sync.Once
	instance types.Store
)

/*
Store is a simple in-memory implementation of that follows
the common store interface. It is an ambient context, so
all sessions derive from the same underlying store instance.
*/
type Store[T any] struct {
	*errnie.Error
	*errnie.State
	data *sync.Map
}

func NewStore[T any]() types.Store {
	once.Do(func() {
		instance = &Store[T]{
			Error: errnie.NewError(),
			data:  new(sync.Map),
		}
	})

	return instance
}

/*
Peek a value from the in-memory store.
*/
func (store *Store[T]) Peek(query *types.Query) (io.Reader, error) {
	if !store.OK() {
		return nil, store.Error
	}

	id, ok := query.Filters["id"]

	if !ok {
		return nil, errnie.Validation(nil, "missing id in filters")
	}

	value, ok := store.data.Load(id)

	if !ok {
		return nil, errnie.Validation(nil, "missing id in filters")
	}

	return value.(*task.Task), nil
}

/*
Poke a value into the in-memory store.
*/
func (store *Store[T]) Poke(query *types.Query) (err error) {
	if !store.OK() {
		return store.Error
	}

	taskItem := task.NewTask()

	if _, err = io.Copy(taskItem, query.Payload); err != nil {
		return errnie.New(errnie.WithError(err))
	}

	store.data.Store(taskItem.ID, taskItem)

	return nil
}
