package errnie

import "sync"

/*
LogController controls errnie logger emission.
*/
type LogController struct {
	lock       sync.Mutex
	suppressed int
}

var defaultLogController = &LogController{}

/*
SuppressLogging disables errnie logging until the returned restore function is
called.
*/
func SuppressLogging() func() {
	return defaultLogController.Suppress()
}

/*
Suppress disables logging for this controller.
*/
func (controller *LogController) Suppress() func() {
	if controller == nil {
		return func() {}
	}

	controller.lock.Lock()
	controller.suppressed++
	controller.lock.Unlock()

	var once sync.Once

	return func() {
		once.Do(func() {
			controller.lock.Lock()
			defer controller.lock.Unlock()

			if controller.suppressed == 0 {
				return
			}

			controller.suppressed--
		})
	}
}

/*
Suppressed reports whether logging is currently disabled.
*/
func (controller *LogController) Suppressed() bool {
	if controller == nil {
		return false
	}

	controller.lock.Lock()
	defer controller.lock.Unlock()

	return controller.suppressed > 0
}

func loggingSuppressed() bool {
	return defaultLogController.Suppressed()
}
