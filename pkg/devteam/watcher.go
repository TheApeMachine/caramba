package devteam

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/lib/pq"
)

const (
	listenChannel  = "kanban_column_change"
	reconnectDelay = 5 * time.Second
)

/*
ColumnEvent is decoded from the pg_notify JSON payload emitted by the
kanban_cards_column_notify trigger.
*/
type ColumnEvent struct {
	ID                string `json:"id"`
	ResearchProjectID string `json:"research_project_id"`
	ColumnKey         string `json:"column_key"`
	OldColumnKey      string `json:"old_column_key"`
	Title             string `json:"title"`
	Description       string `json:"description"`
}

/*
Watcher listens on the Postgres LISTEN/NOTIFY channel and publishes ColumnEvents
on an unbuffered channel. It reconnects automatically on connection failures.
*/
type Watcher struct {
	ctx         context.Context
	databaseURL string
	events      chan ColumnEvent
	errors      chan error
}

/*
NewWatcher constructs a Watcher. Call Watch() to start listening.
*/
func NewWatcher(ctx context.Context, databaseURL string) *Watcher {
	return &Watcher{
		ctx:         ctx,
		databaseURL: databaseURL,
		events:      make(chan ColumnEvent, 64),
		errors:      make(chan error, 64),
	}
}

/*
Events returns the read-only channel of ColumnEvents.
*/
func (watcher *Watcher) Events() <-chan ColumnEvent {
	return watcher.events
}

func (watcher *Watcher) Errors() <-chan error {
	return watcher.errors
}

/*
Watch blocks, maintaining a LISTEN connection and forwarding decoded events.
It returns when ctx is cancelled.
*/
func (watcher *Watcher) Watch() error {
	defer close(watcher.errors)

	for {
		if err := watcher.ctx.Err(); err != nil {
			return nil
		}

		if err := watcher.listenLoop(); err != nil {
			watcher.publishError(err)

			select {
			case <-watcher.ctx.Done():
				return nil
			case <-time.After(reconnectDelay):
			}
		}
	}
}

func (watcher *Watcher) publishError(err error) {
	select {
	case watcher.errors <- err:
	case <-watcher.ctx.Done():
	}
}

func (watcher *Watcher) listenLoop() error {
	listener := pq.NewListener(watcher.databaseURL, reconnectDelay, time.Minute, func(ev pq.ListenerEventType, err error) {
		if err == nil {
			return
		}

		watcher.publishError(fmt.Errorf("watcher: listener event %v: %w", ev, err))
	})

	defer listener.Close()

	if err := listener.Listen(listenChannel); err != nil {
		return fmt.Errorf("watcher: listen: %w", err)
	}

	for {
		select {
		case <-watcher.ctx.Done():
			return nil

		case notification, ok := <-listener.Notify:
			if !ok {
				return fmt.Errorf("watcher: listener channel closed")
			}

			if notification == nil {
				continue
			}

			var event ColumnEvent

			if err := json.Unmarshal([]byte(notification.Extra), &event); err != nil {
				continue
			}

			select {
			case watcher.events <- event:
			case <-watcher.ctx.Done():
				return nil
			}
		}
	}
}
