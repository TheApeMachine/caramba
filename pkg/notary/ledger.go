package notary

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"sync"
)

var (
	ErrInsufficientFunds = errors.New("notary: insufficient credits")
	ErrAccountNotFound   = errors.New("notary: account not found")
	ErrUnauthorizedMint  = errors.New("notary: unauthorized mint")
	ErrInvalidAmount     = errors.New("notary: amount must be positive")
)

/*
Ledger maintains the single source of truth for the Caramba network economy.
It tracks the credit balances of nodes (identified by their public key addresses)
and the hashes of approved Models/Manifests.
*/
type Ledger struct {
	mu            sync.RWMutex
	mintAuthority string
	nonce         uint64
	balances      map[string]int64

	// provenance tracks verified and approved artifacts (models, manifests)
	// Key: Artifact Hash, Value: Owner Address
	provenance map[string]string
}

/*
NewLedger initializes an empty ledger.
*/
func NewLedger(mintAuthority string) *Ledger {
	return &Ledger{
		mintAuthority: mintAuthority,
		balances:      make(map[string]int64),
		provenance:    make(map[string]string),
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
Mint assigns new credits after verifying the mint authority signature.
*/
func (ledger *Ledger) Mint(
	authorityAddress string,
	recipientAddress string,
	amount int64,
	signature []byte,
) error {
	if amount <= 0 {
		return ErrInvalidAmount
	}

	ledger.mu.Lock()
	defer ledger.mu.Unlock()

	if authorityAddress != ledger.mintAuthority {
		return ErrUnauthorizedMint
	}

	nextNonce := ledger.nonce + 1
	payload := MintPayload(authorityAddress, recipientAddress, amount, nextNonce)

	if !Verify(authorityAddress, payload, signature) {
		return ErrInvalidSignature
	}

	ledger.nonce = nextNonce
	ledger.balances[recipientAddress] += amount

	return nil
}

/*
Transfer moves credits from the sender to the recipient.
Returns an error if the sender lacks sufficient funds.
*/
func (ledger *Ledger) Transfer(sender, recipient string, amount int64) error {
	if amount <= 0 {
		return ErrInvalidAmount
	}

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
MintPayload creates the canonical bytes the mint authority signs.
*/
func MintPayload(authorityAddress string, recipientAddress string, amount int64, nonce uint64) []byte {
	hash := sha256.New()
	hash.Write([]byte("caramba:notary:mint:v1"))
	hash.Write([]byte(authorityAddress))
	hash.Write([]byte{0})
	hash.Write([]byte(recipientAddress))
	hash.Write([]byte{0})

	var amountData [8]byte
	binary.BigEndian.PutUint64(amountData[:], uint64(amount))
	hash.Write(amountData[:])

	var nonceData [8]byte
	binary.BigEndian.PutUint64(nonceData[:], nonce)
	hash.Write(nonceData[:])

	return hash.Sum(nil)
}

func (ledger *Ledger) MintAuthority() string {
	ledger.mu.RLock()
	defer ledger.mu.RUnlock()

	return ledger.mintAuthority
}

func (ledger *Ledger) MintNonce() uint64 {
	ledger.mu.RLock()
	defer ledger.mu.RUnlock()

	return ledger.nonce
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
