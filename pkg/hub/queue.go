package hub

import (
	"sync"
	"time"
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
	channels map[string]*Channel
}

func NewQueue() *Queue {
	once.Do(func() {
		instance = &Queue{
			channels: make(map[string]*Channel),
		}
	})
	return instance
}

func (q *Queue) Add(event *Event) {
	q.AddOrDrop(event)
}

func (q *Queue) AddOrDrop(event *Event) {
	channel, ok := q.channels[event.Topic]

	if !ok {
		return
	}

	select {
	case channel.i <- event:
	default:
	}
}

func (q *Queue) Close() {
	for _, channel := range q.channels {
		close(channel.i)
		close(channel.o)
	}
}

func (q *Queue) Subscribe(topic string) <-chan *Event {
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
