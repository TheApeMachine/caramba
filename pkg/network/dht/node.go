package dht

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"

	"github.com/theapemachine/caramba/pkg/notary"
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
	ID              NodeID
	Address         string
	IdentityAddress string
	Profile         ComputeProfile
}

/*
NewNode instantiates a new local node whose Kademlia ID is derived from the
ledger identity public key.
*/
func NewNode(address string, identityAddress string, profile ComputeProfile) (*Node, error) {
	if address == "" {
		return nil, fmt.Errorf("dht: node address is required")
	}

	if _, err := notary.PublicKeyFromAddress(identityAddress); err != nil {
		return nil, err
	}

	return &Node{
		ID:              NewNodeID(identityAddress),
		Address:         address,
		IdentityAddress: identityAddress,
		Profile:         profile,
	}, nil
}

/*
NewObservedNode records a peer announced over the wire after its NodeID bytes
have already passed transport-level validation.
*/
func NewObservedNode(address string, id NodeID, profile ComputeProfile) (*Node, error) {
	if address == "" {
		return nil, fmt.Errorf("dht: node address is required")
	}

	return &Node{
		ID:      id,
		Address: address,
		Profile: profile,
	}, nil
}
