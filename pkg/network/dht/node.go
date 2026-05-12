package dht

import (
	"crypto/sha1"
	"encoding/hex"
)

/*
NodeID represents a 160-bit Kademlia Node ID, which is typically derived
from a SHA-1 hash of the node's public key or address.
*/
type NodeID [20]byte

func NewNodeID(data string) NodeID {
	return sha1.Sum([]byte(data))
}

func (id NodeID) String() string {
	return hex.EncodeToString(id[:])
}

/*
ComputeProfile holds the hardware capabilities of a node, allowing the
Orchestrator to route IR nodes to hardware that can actually run them.
*/
type ComputeProfile struct {
	AvailableRunners []string
	VRAMBytes        uint64
	RAMBytes         uint64
	FLOPPerSec       uint64
}

/*
Node represents a participant in the Caramba distributed research grid.
*/
type Node struct {
	ID      NodeID
	Address string
	Profile ComputeProfile
}

/*
NewNode instantiates a new local node. In a full system, the ID would be
cryptographically tied to the user's wallet/credits.
*/
func NewNode(address string, profile ComputeProfile) (*Node, error) {
	return &Node{
		ID:      NewNodeID(address), // Naive ID generation for now
		Address: address,
		Profile: profile,
	}, nil
}
