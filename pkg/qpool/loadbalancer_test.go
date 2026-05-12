package qpool

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestLoadBalancerObserveAndLimit(t *testing.T) {
	Convey("LoadBalancer limits when queue depth per worker exceeds capacity", t, func() {
		lb := NewLoadBalancer(2, 3)

		So(lb.Limit(), ShouldBeFalse)

		lb.Observe(&MetricReading{WorkerCount: 2, JobQueueSize: 8})

		So(lb.Limit(), ShouldBeTrue)
	})
}

func TestLoadBalancerSelectWorker(t *testing.T) {
	Convey("LoadBalancer SelectWorker returns modulo routing hint", t, func() {
		lb := NewLoadBalancer(3, 5)

		lb.Observe(&MetricReading{WorkerCount: 3, JobQueueSize: 2, TotalJobs: 10})

		id, err := lb.SelectWorker()

		So(err, ShouldBeNil)
		So(id, ShouldEqual, 1)
	})
}
