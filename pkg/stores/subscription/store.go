package subscription

import (
	"errors"
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

type MockStore[T any] struct {
	*errnie.Error
	*errnie.State
	Data          *sync.Map
	Subscriptions map[string][]*Subscription
	Mu            *sync.RWMutex
}

type StoreOption[T any] func(*MockStore[T])

func NewMockStore[T any](opts ...StoreOption[T]) types.Store {
	once.Do(func() {
		instance = &MockStore[T]{
			Error:         errnie.NewError(),
			Data:          new(sync.Map),
			State:         errnie.NewState().To(errnie.StateReady),
			Subscriptions: make(map[string][]*Subscription),
			Mu:            new(sync.RWMutex),
		}
	})

	for _, opt := range opts {
		opt(instance.(*MockStore[T]))
	}

	return instance
}

func (m *MockStore[T]) Peek(query *types.Query) (io.Reader, error) {
	if value, ok := m.Data.Load(query.Filters["id"]); ok {
		if task, ok := value.(*task.Task); ok {
			return task, nil
		}
		return nil, errors.New("value is not a *task.Task")
	}

	return nil, errors.New("not ok")
}

func (m *MockStore[T]) Poke(query *types.Query) (err error) {
	taskItem := task.NewTask()

	if _, err = io.Copy(taskItem, query.Payload); err != nil {
		return errnie.New(errnie.WithError(err))
	}

	m.Data.Store(taskItem.ID, taskItem)

	go func() {
		m.Mu.Lock()
		defer m.Mu.Unlock()

		for _, subscription := range m.Subscriptions[taskItem.ID] {
			if _, err := io.Copy(subscription, taskItem); err != nil {
				errnie.New(errnie.WithError(err))
			}
		}
	}()

	return nil
}

func WithSubscription[T any](
	subscription *Subscription,
) StoreOption[T] {
	return func(store *MockStore[T]) {
		store.Mu.Lock()
		defer store.Mu.Unlock()

		if subscription == nil {
			return
		}

		store.Subscriptions[subscription.ID] = append(
			store.Subscriptions[subscription.ID],
			subscription,
		)
	}
}
