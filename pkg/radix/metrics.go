package radix

import (
	"sync"
	"sync/atomic"
	"time"
)

// Metrics tracks performance and operational metrics for the radix tree
type Metrics struct {
	// Operation counters
	insertCount   atomic.Uint64
	lookupCount   atomic.Uint64
	syncCount     atomic.Uint64
	conflictCount atomic.Uint64

	// Election metrics
	votesReceived atomic.Uint64
	termNumber    atomic.Uint64
	lastVoter     string

	// Latency tracking
	insertLatency  *LatencyTracker
	lookupLatency  *LatencyTracker
	syncLatency    *LatencyTracker
	networkLatency *LatencyTracker

	// Network metrics
	bytesTransmitted atomic.Uint64
	bytesReceived    atomic.Uint64
	peerCount        atomic.Int32

	// Node status
	isLeader     atomic.Bool
	nodeRole     string
	nodeWeight   float64
	lastSyncTime time.Time
	mu           sync.RWMutex
}

// LatencyTracker maintains a rolling window of operation latencies
type LatencyTracker struct {
	window []time.Duration
	mu     sync.RWMutex
	size   int
	pos    int
}

// NewMetrics creates a new metrics tracker
func NewMetrics() *Metrics {
	return &Metrics{
		insertLatency:  NewLatencyTracker(100),
		lookupLatency:  NewLatencyTracker(100),
		syncLatency:    NewLatencyTracker(100),
		networkLatency: NewLatencyTracker(100),
	}
}

// NewLatencyTracker creates a new latency tracker with given window size
func NewLatencyTracker(size int) *LatencyTracker {
	return &LatencyTracker{
		window: make([]time.Duration, size),
		size:   size,
	}
}

// RecordLatency adds a new latency measurement
func (lt *LatencyTracker) RecordLatency(d time.Duration) {
	lt.mu.Lock()
	defer lt.mu.Unlock()
	lt.window[lt.pos] = d
	lt.pos = (lt.pos + 1) % lt.size
}

// AverageLatency returns the average latency over the window
func (lt *LatencyTracker) AverageLatency() time.Duration {
	lt.mu.RLock()
	defer lt.mu.RUnlock()
	var sum time.Duration
	count := 0
	for _, d := range lt.window {
		if d > 0 {
			sum += d
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return sum / time.Duration(count)
}

// RecordInsert records metrics for an insert operation
func (m *Metrics) RecordInsert(duration time.Duration, bytes int) {
	m.insertCount.Add(1)
	m.insertLatency.RecordLatency(duration)
	m.bytesTransmitted.Add(uint64(bytes))
}

// RecordLookup records metrics for a lookup operation
func (m *Metrics) RecordLookup(duration time.Duration) {
	m.lookupCount.Add(1)
	m.lookupLatency.RecordLatency(duration)
}

// RecordSync records metrics for a sync operation
func (m *Metrics) RecordSync(duration time.Duration, bytes int) {
	m.syncCount.Add(1)
	m.syncLatency.RecordLatency(duration)
	m.bytesReceived.Add(uint64(bytes))
	m.mu.Lock()
	m.lastSyncTime = time.Now()
	m.mu.Unlock()
}

// RecordConflict records a detected conflict
func (m *Metrics) RecordConflict() {
	m.conflictCount.Add(1)
}

// UpdatePeerCount updates the current peer count
func (m *Metrics) UpdatePeerCount(count int32) {
	m.peerCount.Store(count)
}

// SetNodeRole updates the node's role
func (m *Metrics) SetNodeRole(role string, weight float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.nodeRole = role
	m.nodeWeight = weight
}

// SetLeader updates the node's leader status
func (m *Metrics) SetLeader(isLeader bool) {
	m.isLeader.Store(isLeader)
}

// RecordVote records a vote received during election
func (m *Metrics) RecordVote(voter string) {
	m.votesReceived.Add(1)
	m.mu.Lock()
	m.lastVoter = voter
	m.mu.Unlock()
}

// GetMetrics returns a snapshot of current metrics
func (m *Metrics) GetMetrics() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return map[string]interface{}{
		"operations": map[string]uint64{
			"insert":   m.insertCount.Load(),
			"lookup":   m.lookupCount.Load(),
			"sync":     m.syncCount.Load(),
			"conflict": m.conflictCount.Load(),
		},
		"election": map[string]interface{}{
			"votes_received": m.votesReceived.Load(),
			"term_number":    m.termNumber.Load(),
			"last_voter":     m.lastVoter,
		},
		"latencies": map[string]float64{
			"insert":  float64(m.insertLatency.AverageLatency()) / float64(time.Millisecond),
			"lookup":  float64(m.lookupLatency.AverageLatency()) / float64(time.Millisecond),
			"sync":    float64(m.syncLatency.AverageLatency()) / float64(time.Millisecond),
			"network": float64(m.networkLatency.AverageLatency()) / float64(time.Millisecond),
		},
		"network": map[string]interface{}{
			"bytes_tx":   m.bytesTransmitted.Load(),
			"bytes_rx":   m.bytesReceived.Load(),
			"peer_count": m.peerCount.Load(),
		},
		"node": map[string]interface{}{
			"is_leader":      m.isLeader.Load(),
			"role":           m.nodeRole,
			"weight":         m.nodeWeight,
			"last_sync_time": m.lastSyncTime,
		},
	}
}
