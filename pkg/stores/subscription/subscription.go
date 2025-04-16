package subscription

import (
	"io"

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
	ID string
	pr *io.PipeReader
	pw *io.PipeWriter
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

func (subscription *Subscription) Read(p []byte) (n int, err error) {
	return subscription.pr.Read(p)
}

func (subscription *Subscription) Write(p []byte) (n int, err error) {
	return subscription.pw.Write(p)
}

func (subscription *Subscription) Close() error {
	if subscription.pr != nil {
		subscription.pr.Close()
	}

	if subscription.pw != nil {
		subscription.pw.Close()
	}

	subscription.ID = ""
	subscription.State.To(errnie.StateDone)
	return nil
}

func WithID(id string) SubscriptionOptions {
	return func(subscription *Subscription) {
		subscription.ID = id
		subscription.pr, subscription.pw = io.Pipe()
	}
}
