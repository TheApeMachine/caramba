package components

const (
	ShowCursor = "\033[?25h"
	HideCursor = "\033[?25l"
)

// RootContainer is the top-level container that manages all components
type RootContainer struct {
	width  int
	height int
}

type RootContainerOption func(*RootContainer)

func NewRootContainer(options ...RootContainerOption) *RootContainer {
	root := &RootContainer{}

	for _, option := range options {
		option(root)
	}

	return root
}

// UpdateSize updates the container dimensions and marks it dirty if changed
func (r *RootContainer) UpdateSize(width, height int) {
	if r.width != width || r.height != height {
		r.width = width
		r.height = height
	}
}

func (r *RootContainer) Read(p []byte) (n int, err error) {
	return r.Read(p)
}

func (r *RootContainer) Write(p []byte) (n int, err error) {
	return r.Write(p)
}

func (r *RootContainer) Close() error {
	return r.Close()
}

func WithSize(width, height int) RootContainerOption {
	return func(r *RootContainer) {
		r.width = width
		r.height = height
	}
}
