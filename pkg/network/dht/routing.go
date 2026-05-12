package dht

import (
	"bytes"
	"math/bits"
	"sort"
)

/*
XORDistance calculates the distance between two Kademlia Node IDs.
The distance is defined as the bitwise XOR of the two IDs.
*/
func XORDistance(a, b NodeID) []byte {
	dist := make([]byte, len(a))
	for i := range a {
		dist[i] = a[i] ^ b[i]
	}
	return dist
}

/*
LeadingZeros counts the number of leading zero bits in a byte slice.
This is used to determine which k-bucket a node belongs to relative
to the local node.
*/
func LeadingZeros(data []byte) int {
	zeros := 0
	for _, b := range data {
		if b == 0 {
			zeros += 8
		} else {
			zeros += bits.LeadingZeros8(b)
			break
		}
	}
	return zeros
}

/*
BucketIndex calculates the k-bucket index for a remote node relative to a local node.
The index is 159 - LeadingZeros(XORDistance(local, remote)).
*/
func BucketIndex(local, remote NodeID) int {
	dist := XORDistance(local, remote)
	lz := LeadingZeros(dist)
	if lz == len(dist)*8 {
		return 0 // Identical nodes go to bucket 0
	}
	return (len(dist) * 8) - 1 - lz
}

/*
NodeDistance tracks a node and its distance from a specific target.
*/
type NodeDistance struct {
	Node     *Node
	Distance []byte
}

/*
SortByDistance sorts a slice of Nodes based on their XOR distance to a target.
*/
func SortByDistance(target NodeID, nodes []*Node) []*Node {
	distances := make([]NodeDistance, len(nodes))
	for i, n := range nodes {
		distances[i] = NodeDistance{
			Node:     n,
			Distance: XORDistance(target, n.ID),
		}
	}

	sort.Slice(distances, func(i, j int) bool {
		return bytes.Compare(distances[i].Distance, distances[j].Distance) < 0
	})

	sorted := make([]*Node, len(nodes))
	for i, d := range distances {
		sorted[i] = d.Node
	}
	return sorted
}
