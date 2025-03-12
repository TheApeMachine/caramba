package workflow

import "io"

type Ring struct {
	slots []io.ReadWriteCloser
	idx   int
}

func NewRing(slots []io.ReadWriteCloser) *Ring {
	return &Ring{
		slots: slots,
		idx:   0,
	}
}

func (ring *Ring) Read(p []byte) (n int, err error) {
	ring.idx++
	return ring.slots[ring.idx%len(ring.slots)].Read(p)
}

func (ring *Ring) Write(p []byte) (n int, err error) {
	ring.idx++
	return ring.slots[ring.idx%len(ring.slots)].Write(p)
}

func (ring *Ring) Close() error {
	ring.idx++
	return ring.slots[ring.idx%len(ring.slots)].Close()
}
