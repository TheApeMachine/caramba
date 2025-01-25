package utils

import (
	"context"
	"errors"
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
)

func TestErrorWithContext(t *testing.T) {
	Convey("Given a base error", t, func() {
		baseErr := errors.New("database connection failed")

		Convey("When creating an error with context", func() {
			errCtx := NewErrorWithContext(baseErr, SeverityHigh).
				WithContext("component", "database").
				WithContext("attempt", 1)

			Convey("Then it should have the expected severity", func() {
				So(errCtx.Severity, ShouldEqual, SeverityHigh)
			})

			Convey("And it should have the expected context values", func() {
				So(errCtx.Context["component"], ShouldEqual, "database")
				So(errCtx.Context["attempt"], ShouldEqual, 1)
			})
		})
	})
}

func TestExponentialBackoff(t *testing.T) {
	Convey("Given an exponential backoff configuration", t, func() {
		backoff := NewExponentialBackoff(
			100*time.Millisecond,
			1*time.Second,
			2.0,
			3,
		)

		Convey("When getting delays for different attempts", func() {
			Convey("The initial delay should be correct", func() {
				So(backoff.NextDelay(0), ShouldEqual, 100*time.Millisecond)
			})

			Convey("The second delay should be exponentially increased", func() {
				So(backoff.NextDelay(1), ShouldEqual, 200*time.Millisecond)
			})

			Convey("The delay should be capped at max delay", func() {
				So(backoff.NextDelay(5), ShouldEqual, 1*time.Second)
			})
		})
	})
}

// Mock implementation of RecoveryStrategy for testing
type mockRecoveryStrategy struct {
	priority   int
	handleFunc func(ctx context.Context, err *ErrorWithContext) error
}

func (m *mockRecoveryStrategy) HandleError(ctx context.Context, err *ErrorWithContext) error {
	if m.handleFunc != nil {
		return m.handleFunc(ctx, err)
	}
	return nil
}

func (m *mockRecoveryStrategy) Cleanup(ctx context.Context) error {
	return nil
}

func (m *mockRecoveryStrategy) Priority() int {
	return m.priority
}

func TestRecoveryManager(t *testing.T) {
	Convey("Given a recovery manager with multiple strategies", t, func() {
		ctx := context.Background()
		rm := NewRecoveryManager()

		strategy1 := &mockRecoveryStrategy{
			priority: 1,
			handleFunc: func(ctx context.Context, err *ErrorWithContext) error {
				return errors.New("strategy 1 failed")
			},
		}

		strategy2 := &mockRecoveryStrategy{
			priority: 2,
			handleFunc: func(ctx context.Context, err *ErrorWithContext) error {
				return nil // This strategy succeeds
			},
		}

		rm.AddStrategy(SeverityHigh, strategy1)
		rm.AddStrategy(SeverityHigh, strategy2)

		Convey("When handling an error", func() {
			testErr := NewErrorWithContext(errors.New("test error"), SeverityHigh)
			err := rm.HandleError(ctx, testErr)

			Convey("Then it should successfully recover", func() {
				So(err, ShouldBeNil)
			})
		})
	})
}

// Mock health monitor implementation
type mockMonitor struct{}

func (m *mockMonitor) CheckHealth() HealthStatus {
	return StatusHealthy
}

func (m *mockMonitor) GetMetrics() map[string]interface{} {
	return map[string]interface{}{
		"uptime": "10m",
	}
}

func TestHealthMonitoring(t *testing.T) {
	Convey("Given a recovery manager with a health monitor", t, func() {
		rm := NewRecoveryManager()
		rm.AddMonitor("test-component", &mockMonitor{})

		Convey("When checking health status", func() {
			status := rm.GetHealthStatus()

			Convey("Then the component should be healthy", func() {
				So(status["test-component"], ShouldEqual, StatusHealthy)
			})
		})
	})
}
