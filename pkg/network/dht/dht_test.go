package dht

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/notary"
)

func TestDHTRouting(t *testing.T) {
	Convey("Given a Local Node and a Routing Table", t, func() {
		localProfile := ComputeProfile{
			AvailableRunners: []string{"cpu"},
			RAMBytes:         8 * 1024 * 1024 * 1024,
		}
		localIdentity, err := notary.NewIdentity()
		So(err, ShouldBeNil)
		localNode, err := NewNode("127.0.0.1:8000", localIdentity.Address(), localProfile)
		So(err, ShouldBeNil)
		So(localNode, ShouldNotBeNil)
		So(localNode.ID, ShouldEqual, NewNodeID(localIdentity.Address()))

		dhtInstance := NewDHT(localNode, func(ctx context.Context, target *Node) bool { return true })

		Convey("When XORDistance is calculated", func() {
			a := NewNodeID("node_A")
			b := NewNodeID("node_B")
			dist := XORDistance(a, b)

			Convey("It should correctly compute the bitwise distance", func() {
				So(len(dist), ShouldEqual, 20)
				// Distance to self should be 0
				distToSelf := XORDistance(a, a)
				for _, b := range distToSelf {
					So(b, ShouldEqual, 0)
				}
			})
		})

		Convey("When adding remote nodes to the DHT", func() {
			remoteIdentity1, err := notary.NewIdentity()
			So(err, ShouldBeNil)
			remoteIdentity2, err := notary.NewIdentity()
			So(err, ShouldBeNil)
			remoteIdentity3, err := notary.NewIdentity()
			So(err, ShouldBeNil)

			remote1, _ := NewNode("10.0.0.1:8001", remoteIdentity1.Address(), ComputeProfile{
				AvailableRunners: []string{"cuda", "cpu"},
			})
			remote2, _ := NewNode("10.0.0.2:8002", remoteIdentity2.Address(), ComputeProfile{
				AvailableRunners: []string{"metal", "cpu"},
			})
			remote3, _ := NewNode("10.0.0.3:8003", remoteIdentity3.Address(), ComputeProfile{
				AvailableRunners: []string{"xla"},
			})

			dhtInstance.AddNode(remote1)
			dhtInstance.AddNode(remote2)
			dhtInstance.AddNode(remote3)

			Convey("It should store the nodes in the routing table", func() {
				// Search for nodes closest to the local node
				closest := dhtInstance.FindClosest(localNode.ID, 10)
				So(len(closest), ShouldEqual, 3)
			})

			Convey("It should allow capability-based routing lookups", func() {
				ctx := context.Background()

				cudaNodes := dhtInstance.LookupHardware(ctx, "cuda", 10)
				So(len(cudaNodes), ShouldEqual, 1)
				So(cudaNodes[0].ID, ShouldEqual, remote1.ID)

				metalNodes := dhtInstance.LookupHardware(ctx, "metal", 10)
				So(len(metalNodes), ShouldEqual, 1)
				So(metalNodes[0].ID, ShouldEqual, remote2.ID)

				cpuNodes := dhtInstance.LookupHardware(ctx, "cpu", 10)
				So(len(cpuNodes), ShouldEqual, 2) // remote1 and remote2
			})
		})
	})
}
