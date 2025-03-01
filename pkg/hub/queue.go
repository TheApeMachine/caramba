package hub

import (
	"fmt"
	"sync"
	"time"

	"github.com/theapemachine/caramba/pkg/output"
)

var once sync.Once
var instance *Queue

type Channel struct {
	i chan *Event
	o chan *Event
}

func NewChannel() *Channel {
	return &Channel{
		i: make(chan *Event, 100),
		o: make(chan *Event, 100),
	}
}

type Queue struct {
	logger   *output.Logger
	channels map[string]*Channel
}

func NewQueue() *Queue {
	once.Do(func() {
		instance = &Queue{
			logger:   output.NewLogger(),
			channels: make(map[string]*Channel),
		}
	})
	return instance
}

func (q *Queue) Add(event *Event) {
	q.AddOrDrop(event)
}

func (q *Queue) AddOrDrop(event *Event) {
	// Check if channel exists for this topic
	channel, ok := q.channels[event.Topic]
	if !ok {
		q.logger.Log(fmt.Sprintf("Dropping event: %s (%s - no channel)", event.String(), event.Topic))
		return
	}

	select {
	case channel.i <- event:
	default:
		q.logger.Log(fmt.Sprintf("Dropping event: %s (%s - channel full)", event.String(), event.Topic))
	}
}

func (q *Queue) Close() {
	for _, channel := range q.channels {
		close(channel.i)
		close(channel.o)
	}
}

func (q *Queue) Subscribe(topic string) <-chan *Event {
	q.logger.Log(fmt.Sprintf("Subscribing to topic: %s", topic))
	channel, ok := q.channels[topic]

	if !ok {
		channel = NewChannel()
		q.channels[topic] = channel

		go func() {
			for msg := range channel.i {
				select {
				case channel.o <- msg:
				default:
					time.Sleep(100 * time.Millisecond)
				}
			}
		}()
	}

	return channel.o
}
