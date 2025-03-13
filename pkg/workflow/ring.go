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
	// Increment index to move to next component
	ring.idx++

	// Get the current component to read from
	currentIdx := ring.idx % len(ring.slots)

	// Read from the current component
	n, err = ring.slots[currentIdx].Read(p)

	// If we got EOF, we should stop the ring rotation
	// This prevents infinite loops when components have no more data
	if err == io.EOF {
		return n, io.EOF
	}

	return n, err
}

func (ring *Ring) Write(p []byte) (n int, err error) {
	ring.idx++
	return ring.slots[ring.idx%len(ring.slots)].Write(p)
}

func (ring *Ring) Close() error {
	ring.idx++
	return ring.slots[ring.idx%len(ring.slots)].Close()
}
