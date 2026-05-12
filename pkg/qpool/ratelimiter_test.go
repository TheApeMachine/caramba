package qpool

import (
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
)

func TestRateLimiter_NewRateLimiter_burstAndRefillInterval(t *testing.T) {
	Convey("Given NewRateLimiter with burst 2 and 50ms refill", t, func() {
		rl := NewRateLimiter(2, 50*time.Millisecond)

		Convey("It should use at least the configured refill interval internally", func() {
			So(rl.refillRate, ShouldEqual, 50*time.Millisecond)
			So(rl.maxTokens, ShouldEqual, int64(2))
		})

		Convey("It should allow two immediate Limit acquisitions then reject until refill", func() {
			So(rl.Limit(), ShouldBeFalse)
			So(rl.Limit(), ShouldBeFalse)
			So(rl.Limit(), ShouldBeTrue)

			time.Sleep(250 * time.Millisecond)

			So(rl.Limit(), ShouldBeFalse)
		})
	})
}

func TestRateLimiter_NewRateLimiter_clampsNegativeCapacity(t *testing.T) {
	Convey("Given NewRateLimiter with negative maxTokens", t, func() {
		rl := NewRateLimiter(-3, 100*time.Millisecond)

		Convey("It should clamp capacity to zero so Limit rejects immediately", func() {
			So(rl.maxTokens, ShouldEqual, 0)
			So(rl.Limit(), ShouldBeTrue)
		})
	})
}

func TestRateLimiter_NewRateLimiter_defaultRefillWhenNonPositive(t *testing.T) {
	Convey("Given NewRateLimiter with non-positive refill duration", t, func() {
		rl := NewRateLimiter(1, 0)

		Convey("It should substitute a positive default refill interval", func() {
			So(rl.refillRate, ShouldEqual, time.Second)
		})
	})
}

func TestRateLimiter_Observe_doesNotRefillTokens(t *testing.T) {
	Convey("Given an exhausted RateLimiter", t, func() {
		rl := NewRateLimiter(1, time.Minute)
		So(rl.Limit(), ShouldBeFalse)
		So(rl.Limit(), ShouldBeTrue)

		Convey("Observe should not add tokens by itself", func() {
			rl.Observe(&MetricReading{TotalJobs: 100, FailedJobs: 2, ThrottledJobs: 1})
			So(rl.Limit(), ShouldBeTrue)
		})
	})
}

func TestRateLimiter_Renormalize_andRefill(t *testing.T) {
	Convey("Given a RateLimiter exhausted by Limit calls", t, func() {
		rl := NewRateLimiter(1, 50*time.Millisecond)
		So(rl.Limit(), ShouldBeFalse)
		So(rl.Limit(), ShouldBeTrue)

		Convey("Renormalize after waiting the refill interval should allow another acquisition", func() {
			time.Sleep(250 * time.Millisecond)
			rl.Renormalize()
			So(rl.Limit(), ShouldBeFalse)
		})
	})
}
