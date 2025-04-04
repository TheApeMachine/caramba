package core

type Status uint

const (
	StatusUnknown Status = iota
	StatusReady
	StatusWaiting
	StatusWorking
	StatusDone
	StatusError
	StatusBusy
)

func (status Status) String() string {
	return []string{
		"unknown",
		"ready",
		"waiting",
		"working",
		"done",
		"error",
		"busy",
	}[status]
}
