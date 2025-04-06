package layouts

import (
	"io"
)

// Layout is an interface for objects that can arrange components
type Layout interface {
	io.ReadWriteCloser
	ContextAware

	// Positioning
	SetRect(rect Rect)
}
