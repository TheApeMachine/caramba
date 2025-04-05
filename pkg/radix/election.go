package radix

import (
	"math/rand"
	"sync"
	"time"
)

// NodeState represents the current state of a node in the election process
type NodeState int

const (
	Follower NodeState = iota
	Candidate
	Leader
)

// ElectionConfig holds configuration for leader election
type ElectionConfig struct {
	// Base timeout for elections (will be randomized)
	ElectionTimeout time.Duration
	// How often to send heartbeats when leader
	HeartbeatInterval time.Duration
	// Minimum number of nodes needed for election
	QuorumSize int
}

// Election manages the leader election process
type Election struct {
	config ElectionConfig
	node   *NetworkNode

	// Election state
	state     NodeState
	term      uint64
	votedFor  string
	stateLock sync.RWMutex

	// Election timers
	electionTimer  *time.Timer
	heartbeatTimer *time.Timer

	// Control channels
	votes    chan string
	shutdown chan struct{}
}

// NewElection creates a new election manager
func NewElection(config ElectionConfig, node *NetworkNode) *Election {
	e := &Election{
		config:   config,
		node:     node,
		state:    Follower,
		votes:    make(chan string, 100),
		shutdown: make(chan struct{}),
	}

	// Start election management
	go e.run()

	return e
}

// run manages the election state machine
func (e *Election) run() {
	e.resetElectionTimer()

	for {
		select {
		case <-e.shutdown:
			return

		case <-e.electionTimer.C:
			e.startElection()

		case voter := <-e.votes:
			e.handleVote(voter)

		case <-e.heartbeatTimer.C:
			if e.getState() == Leader {
				e.sendHeartbeats()
			}
		}
	}
}

// startElection initiates a new election
func (e *Election) startElection() {
	e.stateLock.Lock()
	e.state = Candidate
	e.term++
	e.votedFor = e.node.config.NodeID
	currentTerm := e.term
	e.stateLock.Unlock()

	// Update metrics
	e.node.metrics.SetLeader(false)

	// Request votes from all peers
	e.node.peersMutex.RLock()
	peers := make([]*peer, 0, len(e.node.peers))
	for _, p := range e.node.peers {
		peers = append(peers, p)
	}
	e.node.peersMutex.RUnlock()

	// Track votes received (including self-vote)
	votesReceived := 1
	votesNeeded := (len(peers) / 2) + 1

	// Request votes from all peers
	for _, p := range peers {
		go func(peer *peer) {
			future, release := peer.client.RequestVote(e.node.ctx, func(p RadixRPC_requestVote_Params) error {
				p.SetTerm(currentTerm)
				p.SetCandidateId(e.node.config.NodeID)
				p.SetLastLogIndex(uint64(len(e.node.merkleTree.Root.Hash)))
				return nil
			})
			defer release()

			result, err := future.Struct()
			if err != nil {
				return
			}

			if result.VoteGranted() {
				e.votes <- peer.addr
			}
		}(p)
	}

	// Wait for votes or timeout
	timeout := time.After(e.config.ElectionTimeout)
	for votesReceived < votesNeeded {
		select {
		case <-e.votes:
			votesReceived++
		case <-timeout:
			return
		case <-e.shutdown:
			return
		}
	}

	// Won election
	if votesReceived >= votesNeeded {
		e.becomeLeader()
	}
}

// becomeLeader transitions the node to leader state
func (e *Election) becomeLeader() {
	e.stateLock.Lock()
	e.state = Leader
	e.stateLock.Unlock()

	// Update metrics
	e.node.metrics.SetLeader(true)

	// Start heartbeat timer
	e.heartbeatTimer = time.NewTimer(e.config.HeartbeatInterval)
}

// sendHeartbeats sends heartbeat messages to all peers
func (e *Election) sendHeartbeats() {
	e.node.peersMutex.RLock()
	peers := make([]*peer, 0, len(e.node.peers))
	for _, p := range e.node.peers {
		peers = append(peers, p)
	}
	e.node.peersMutex.RUnlock()

	for _, p := range peers {
		go func(peer *peer) {
			future, release := peer.client.Heartbeat(e.node.ctx, func(p RadixRPC_heartbeat_Params) error {
				p.SetTerm(e.term)
				p.SetLeaderId(e.node.config.NodeID)
				return nil
			})
			defer release()

			result, err := future.Struct()
			if err != nil {
				return
			}

			// Step down if peer has higher term
			if result.Term() > e.term {
				e.stepDown(result.Term())
			}
		}(p)
	}

	// Reset heartbeat timer
	e.heartbeatTimer.Reset(e.config.HeartbeatInterval)
}

// stepDown steps down from leader/candidate to follower
func (e *Election) stepDown(newTerm uint64) {
	e.stateLock.Lock()
	e.state = Follower
	e.term = newTerm
	e.votedFor = ""
	e.stateLock.Unlock()

	// Update metrics
	e.node.metrics.SetLeader(false)

	// Reset election timer
	e.resetElectionTimer()
}

// resetElectionTimer resets the election timeout with random jitter
func (e *Election) resetElectionTimer() {
	if e.electionTimer != nil {
		e.electionTimer.Stop()
	}

	// Add random jitter to election timeout
	jitter := time.Duration(rand.Int63n(int64(e.config.ElectionTimeout)))
	timeout := e.config.ElectionTimeout + jitter

	e.electionTimer = time.NewTimer(timeout)
}

// getState returns the current node state
func (e *Election) getState() NodeState {
	e.stateLock.RLock()
	defer e.stateLock.RUnlock()
	return e.state
}

// handleVote processes a vote received from a peer
func (e *Election) handleVote(voter string) {
	e.stateLock.Lock()
	defer e.stateLock.Unlock()

	// Only count votes if still a candidate
	if e.state != Candidate {
		return
	}

	// Record vote in metrics
	e.node.metrics.RecordVote(voter)
}

// handleVoteRequest processes a vote request from a candidate
func (e *Election) handleVoteRequest(term uint64, candidateId string, lastLogIndex uint64) bool {
	e.stateLock.Lock()
	defer e.stateLock.Unlock()

	// Step down if term is higher
	if term > e.term {
		e.stepDown(term)
	}

	// Don't grant vote if candidate's term is lower
	if term < e.term {
		return false
	}

	// Only vote if we haven't voted for anyone else in this term
	// or if we've already voted for this candidate
	if e.votedFor == "" || e.votedFor == candidateId {
		// Verify the candidate's log is at least as up-to-date as ours
		if lastLogIndex >= uint64(len(e.node.merkleTree.Root.Hash)) {
			e.votedFor = candidateId
			e.term = term
			return true
		}
	}

	return false
}

// handleHeartbeat processes a heartbeat from the leader
func (e *Election) handleHeartbeat(term uint64, leaderId string) bool {
	e.stateLock.Lock()
	defer e.stateLock.Unlock()

	// Step down if term is higher
	if term > e.term {
		e.stepDown(term)
		// Update metrics with new leader
		e.node.metrics.SetNodeRole("follower", 0.0)
		return true
	}

	// Reject if term is lower
	if term < e.term {
		return false
	}

	// Accept heartbeat if term matches and from valid leader
	if e.state != Leader && leaderId != "" {
		e.resetElectionTimer()
		// Update metrics
		e.node.metrics.termNumber.Store(term)
		e.node.metrics.SetNodeRole("follower", 0.0)
		return true
	}

	return false
}

// Close shuts down the election manager
func (e *Election) Close() {
	close(e.shutdown)
}
