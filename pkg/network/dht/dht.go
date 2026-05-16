package dht

import (
	"context"
	"sync"
)

const (
	// K represents the maximum number of nodes stored in each k-bucket.
	K = 20
	// IDBits is the number of bits in the Node ID.
	IDBits = 160
)

/*
PingFunc checks if a remote node is alive and responsive.
*/
type PingFunc func(ctx context.Context, target *Node) bool

/*
KBucket represents a single bucket in the Kademlia routing table.
It stores up to K nodes.
*/
type KBucket struct {
	mu       sync.RWMutex
	nodes    []*Node
	pingFunc PingFunc
}

func NewKBucket(ping PingFunc) *KBucket {
	return &KBucket{
		nodes:    make([]*Node, 0, K),
		pingFunc: ping,
	}
}

/*
Update adds a node to the bucket or moves it to the end if it already exists,
following Kademlia's least-recently-seen eviction policy.
*/
func (bucket *KBucket) Update(node *Node) {
	bucket.mu.Lock()

	// Check if node exists
	for i, n := range bucket.nodes {
		if n.ID == node.ID {
			// Move to end (most recently seen)
			bucket.nodes = append(bucket.nodes[:i], bucket.nodes[i+1:]...)
			bucket.nodes = append(bucket.nodes, node)
			bucket.mu.Unlock()
			return
		}
	}

	// If not full, simply append
	if len(bucket.nodes) < K {
		bucket.nodes = append(bucket.nodes, node)
		bucket.mu.Unlock()
		return
	}

	// The bucket is full. Get the least recently seen node (index 0).
	leastRecentlySeen := bucket.nodes[0]
	bucket.mu.Unlock() // Unlock to perform network I/O

	// Ping the least recently seen node synchronously while the bucket lock is released.
	alive := false
	if bucket.pingFunc != nil {
		alive = bucket.pingFunc(context.Background(), leastRecentlySeen)
	}

	bucket.mu.Lock()
	defer bucket.mu.Unlock()

	if alive {
		// The node is alive. It is moved to the tail. The new node is discarded.
		// We must find it again because the slice might have changed while unlocked.
		for i, n := range bucket.nodes {
			if n.ID == leastRecentlySeen.ID {
				bucket.nodes = append(bucket.nodes[:i], bucket.nodes[i+1:]...)
				bucket.nodes = append(bucket.nodes, leastRecentlySeen)
				break
			}
		}
	} else {
		// The node is dead. We evict it and insert the new node at the tail.
		for i, n := range bucket.nodes {
			if n.ID == leastRecentlySeen.ID {
				bucket.nodes = append(bucket.nodes[:i], bucket.nodes[i+1:]...)
				bucket.nodes = append(bucket.nodes, node)
				break
			}
		}
	}
}

/*
Nodes returns a copy of the nodes currently in the bucket.
*/
func (bucket *KBucket) Nodes() []*Node {
	bucket.mu.RLock()
	defer bucket.mu.RUnlock()
	out := make([]*Node, len(bucket.nodes))
	copy(out, bucket.nodes)
	return out
}

/*
RoutingTable implements a Kademlia routing table.
*/
type RoutingTable struct {
	local   *Node
	buckets [IDBits]*KBucket
}

/*
NewRoutingTable creates a new routing table for the local node.
*/
func NewRoutingTable(local *Node, ping PingFunc) *RoutingTable {
	rt := &RoutingTable{
		local: local,
	}
	for i := range rt.buckets {
		rt.buckets[i] = NewKBucket(ping)
	}
	return rt
}

/*
Update adds or refreshes a node in the appropriate k-bucket.
*/
func (rt *RoutingTable) Update(node *Node) {
	if node.ID == rt.local.ID {
		return
	}
	index := BucketIndex(rt.local.ID, node.ID)
	if index >= 0 && index < IDBits {
		rt.buckets[index].Update(node)
	}
}

/*
FindClosestNodes returns up to count nodes closest to the target ID.
*/
func (rt *RoutingTable) FindClosestNodes(target NodeID, count int) []*Node {
	var allNodes []*Node
	// Collect nodes from all buckets.
	// Optimization: start from the bucket corresponding to the target's distance
	// and expand outwards. For simplicity, we collect all and sort.
	for _, bucket := range rt.buckets {
		allNodes = append(allNodes, bucket.Nodes()...)
	}

	sorted := SortByDistance(target, allNodes)
	if len(sorted) > count {
		return sorted[:count]
	}
	return sorted
}

/*
DHT represents the high-level Distributed Hash Table instance for a node.
*/
type DHT struct {
	local   *Node
	routing *RoutingTable

	// Value store for saving arbitrary data
	storeMu sync.RWMutex
	store   map[string][]byte
}

func NewDHT(local *Node, ping PingFunc) *DHT {
	return &DHT{
		local:   local,
		routing: NewRoutingTable(local, ping),
		store:   make(map[string][]byte),
	}
}

func (dht *DHT) LocalNode() *Node {
	return dht.local
}

func (dht *DHT) AddNode(node *Node) {
	dht.routing.Update(node)
}

func (dht *DHT) FindClosest(target NodeID, count int) []*Node {
	return dht.routing.FindClosestNodes(target, count)
}

// StoreValue saves a value in the local DHT store.
func (dht *DHT) StoreValue(key string, value []byte) {
	dht.storeMu.Lock()
	defer dht.storeMu.Unlock()
	dht.store[key] = value
}

// GetValue retrieves a value from the local DHT store.
func (dht *DHT) GetValue(key string) ([]byte, bool) {
	dht.storeMu.RLock()
	defer dht.storeMu.RUnlock()
	val, ok := dht.store[key]
	return val, ok
}

/*
LookupHardware queries the local routing table for nodes that match specific hardware profiles.
*/
func (dht *DHT) LookupHardware(ctx context.Context, requirement string, count int) []*Node {
	var matches []*Node

	// Scan all buckets for nodes matching the required capability
	for _, bucket := range dht.routing.buckets {
		for _, node := range bucket.Nodes() {
			for _, runner := range node.Profile.AvailableRunners {
				if runner == requirement {
					matches = append(matches, node)
					if len(matches) >= count {
						return matches
					}
					break
				}
			}
		}
	}

	return matches
}
