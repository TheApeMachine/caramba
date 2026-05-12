package qpool

import (
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
)

func TestCircuitBreakerRegulatorInterface(t *testing.T) {
	Convey("Given a CircuitBreaker", t, func() {
		breaker := NewCircuitBreaker(2, 100*time.Millisecond, 2)

		Convey("It should satisfy the Regulator interface", func() {
			var _ Regulator = breaker
		})

		Convey("It should not limit when observed with empty metrics", func() {
			breaker.Observe(&MetricReading{})
			So(breaker.Limit(), ShouldBeFalse)
		})
	})
}

func TestCircuitBreakerOpensAfterFailures(t *testing.T) {
	Convey("breaker opens then permits probe after timeout", t, func() {
		breaker := NewCircuitBreaker(2, 100*time.Millisecond, 2)

		So(breaker.state.Load(), ShouldEqual, cbClosed)

		breaker.RecordFailure()
		breaker.RecordFailure()

		So(breaker.Allow(), ShouldBeFalse)
		So(breaker.state.Load(), ShouldEqual, cbOpen)

		time.Sleep(150 * time.Millisecond)

		So(breaker.Allow(), ShouldBeTrue)
		So(breaker.state.Load(), ShouldEqual, cbHalfOpen)
	})
}

func TestCircuitBreakerHalfOpenClosesAfterSuccesses(t *testing.T) {
	Convey("half-open collects successes then closes", t, func() {
		breaker := NewCircuitBreaker(2, 100*time.Millisecond, 2)

		breaker.RecordFailure()
		breaker.RecordFailure()

		time.Sleep(150 * time.Millisecond)

		/* Half-open allows up to halfOpenMax concurrent probes; two Allow() calls are expected. */
		So(breaker.Allow(), ShouldBeTrue)
		So(breaker.Allow(), ShouldBeTrue)

		breaker.RecordSuccess()

		So(breaker.state.Load(), ShouldEqual, cbHalfOpen)

		breaker.RecordSuccess()

		So(breaker.state.Load(), ShouldEqual, cbClosed)
	})
}

func TestCircuitBreakerRenormalize(t *testing.T) {
	Convey("Renormalize moves open breaker toward half-open", t, func() {
		breaker := NewCircuitBreaker(2, 100*time.Millisecond, 2)

		breaker.RecordFailure()
		breaker.RecordFailure()

		So(breaker.state.Load(), ShouldEqual, cbOpen)

		time.Sleep(150 * time.Millisecond)

		breaker.Renormalize()

		So(breaker.state.Load(), ShouldEqual, cbHalfOpen)
		So(breaker.halfOpenSuccess.Load(), ShouldEqual, 0)
	})
}
