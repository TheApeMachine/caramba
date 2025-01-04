package drknow

import "time"

/*
Perspective represents a single view or response, and is carried directly
by the owner of the perspective.
*/
type Perspective struct {
	ID         string
	Owner      string
	Content    interface{}
	Confidence float64
	Reasoning  []string
	Method     string
	Timestamp  time.Time
}
