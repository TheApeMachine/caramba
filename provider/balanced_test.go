package provider

import (
	"testing"
	"time"

	. "github.com/smartystreets/goconvey/convey"
)

func TestBalancedProvider(t *testing.T) {
	Convey("Given a BalancedProvider", t, func() {
		bp := &BalancedProvider{
			providers: []*ProviderStatus{
				{
					name:     "test-provider-1",
					occupied: false,
					failures: 0,
					lastUsed: time.Now().Add(-2 * time.Hour),
				},
				{
					name:     "test-provider-2",
					occupied: false,
					failures: 2,
					lastUsed: time.Now().Add(-1 * time.Hour),
				},
				{
					name:     "test-provider-3",
					occupied: true,
					failures: 0,
					lastUsed: time.Now(),
				},
			},
		}

		Convey("When checking provider availability", func() {
			cooldownPeriod := 60 * time.Second
			maxFailures := 3

			Convey("Should identify available providers correctly", func() {
				available := bp.isProviderAvailable(bp.providers[0], cooldownPeriod, maxFailures)
				So(available, ShouldBeTrue)

				occupied := bp.isProviderAvailable(bp.providers[2], cooldownPeriod, maxFailures)
				So(occupied, ShouldBeFalse)
			})
		})

		Convey("When comparing providers", func() {
			oldestUse := time.Now()

			Convey("Should prefer providers with fewer failures", func() {
				better := bp.isBetterProvider(bp.providers[0], bp.providers[1], oldestUse)
				So(better, ShouldBeTrue)
			})
		})

		Convey("When getting unoccupied providers", func() {
			unoccupied := bp.getUnoccupiedProviders()

			Convey("Should return correct number of available providers", func() {
				So(len(unoccupied), ShouldEqual, 2)
			})

			Convey("Should not include occupied providers", func() {
				for _, p := range unoccupied {
					So(p.occupied, ShouldBeFalse)
				}
			})
		})

		Convey("When selecting best provider", func() {
			best := bp.selectBestProvider()

			Convey("Should select provider with fewest failures and oldest last use", func() {
				So(best, ShouldNotBeNil)
				So(best.name, ShouldEqual, "test-provider-1")
			})

			Convey("Should mark selected provider as occupied", func() {
				So(best.occupied, ShouldBeTrue)
				So(time.Since(best.lastUsed), ShouldBeLessThan, time.Second)
			})
		})
	})
}
