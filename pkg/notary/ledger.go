package notary

import (
	"errors"
	"sync"
)

var (
	ErrInsufficientFunds = errors.New("notary: insufficient credits")
	ErrAccountNotFound   = errors.New("notary: account not found")
)

/*
Ledger maintains the single source of truth for the Caramba network economy.
It tracks the credit balances of nodes (identified by their public key addresses)
and the hashes of approved Models/Manifests.
*/
type Ledger struct {
	mu       sync.RWMutex
	balances map[string]int64

	// provenance tracks verified and approved artifacts (models, manifests)
	// Key: Artifact Hash, Value: Owner Address
	provenance map[string]string
}

/*
NewLedger initializes an empty ledger.
*/
func NewLedger() *Ledger {
	return &Ledger{
		balances:   make(map[string]int64),
		provenance: make(map[string]string),
	}
}

/*
BalanceOf returns the current credit balance of a given address.
*/
func (ledger *Ledger) BalanceOf(address string) int64 {
	ledger.mu.RLock()
	defer ledger.mu.RUnlock()
	return ledger.balances[address]
}

/*
Mint creates new credits out of thin air and assigns them to an address.
In a real decentralized system, this would be strictly controlled by consensus.
*/
func (ledger *Ledger) Mint(address string, amount int64) {
	ledger.mu.Lock()
	defer ledger.mu.Unlock()
	ledger.balances[address] += amount
}

/*
Transfer moves credits from the sender to the recipient.
Returns an error if the sender lacks sufficient funds.
*/
func (ledger *Ledger) Transfer(sender, recipient string, amount int64) error {
	ledger.mu.Lock()
	defer ledger.mu.Unlock()

	senderBalance := ledger.balances[sender]
	if senderBalance < amount {
		return ErrInsufficientFunds
	}

	ledger.balances[sender] -= amount
	ledger.balances[recipient] += amount
	return nil
}

/*
RecordArtifact stores a cryptographic hash of a model or manifest, associating
it with its creator's address. This is the "chain of custody" step.
*/
func (ledger *Ledger) RecordArtifact(artifactHash string, ownerAddress string) {
	ledger.mu.Lock()
	defer ledger.mu.Unlock()
	ledger.provenance[artifactHash] = ownerAddress
}

/*
VerifyArtifact checks if an artifact hash exists in the ledger and belongs
to the specified owner.
*/
func (ledger *Ledger) VerifyArtifact(artifactHash string, ownerAddress string) bool {
	ledger.mu.RLock()
	defer ledger.mu.RUnlock()
	owner, exists := ledger.provenance[artifactHash]
	return exists && owner == ownerAddress
}
