package types

import (
	"bytes"
	"io"
	"sync"

	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Subscription is a way for a component to receive updates and notifications
from the store. It is a read-only interface that allows components to
receive data from the store without modifying it. It is used to
subscribe to changes in the store and receive updates when the data
changes. It is a way to decouple components and allow them to
receive updates without being tightly coupled to the store.
*/
type Subscription struct {
	*errnie.Error
	*errnie.State
	store       Store
	subscribers *sync.Map
}

type SubscriptionOptions func(*Subscription)

func NewSubscription(opts ...SubscriptionOptions) *Subscription {
	subscription := &Subscription{
		Error: errnie.NewError(),
		State: errnie.NewState().To(errnie.StateReady),
	}

	for _, opt := range opts {
		opt(subscription)
	}

	return subscription
}

/*
Subscribe to an item in a store.
*/
func (subscription *Subscription) Subscribe(id string) (io.Reader, error) {
	if !subscription.OK() {
		return nil, subscription.Error
	}

	buf := bytes.NewBuffer([]byte{})
	subscription.subscribers.Store(id, buf)

	return buf, nil
}

/*
Unsubscribe from an item in a store.
*/
func (subscription *Subscription) Unsubscribe(id string) error {
	if !subscription.OK() {
		return subscription.Error
	}

	subscription.subscribers.Delete(id)

	return nil
}

/*
Notify all subscribers of an item in a store.
*/
func (subscription *Subscription) Notify(id string, data []byte) error {
	if !subscription.OK() {
		return subscription.Error
	}

	if data == nil {
		return errnie.Validation(nil, "data is nil")
	}

	buf, ok := subscription.subscribers.Load(id)

	if !ok {
		return errnie.Validation(nil, "subscriber not found")
	}

	_, err := buf.(*bytes.Buffer).Write(data)

	if err != nil {
		return errnie.Validation(err, "failed to write data to subscriber")
	}

	return nil
}

func WithStore(store Store) SubscriptionOptions {
	return func(subscription *Subscription) {
		subscription.store = store
	}
}
