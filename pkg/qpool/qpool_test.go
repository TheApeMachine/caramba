package qpool

import (
	"context"
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
)

func TestQPoolScheduleSimple(t *testing.T) {
	Convey("Given a new Q pool", t, func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)

		q := NewQ(ctx, 2, 5, &Config{
			SchedulingTimeout: time.Second,
			Scaler:            nil,
		})

		Reset(func() {
			cancel()

			if q != nil {
				q.Close()
			}
		})

		Convey("When scheduling a simple job", func() {
			resultCh := q.Schedule("test-job", func(ctx context.Context) (any, error) {
				return "success", nil
			})

			select {
			case <-ctx.Done():
				So(ctx.Err(), ShouldBeNil)
			case result := <-resultCh:
				So(result, ShouldNotBeNil)
				So(result.Error, ShouldBeNil)
				So(result.Value, ShouldEqual, "success")
			}
		})
	})
}
