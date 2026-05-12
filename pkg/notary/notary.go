package notary

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
)

var (
	ErrInvalidSignature = errors.New("notary: invalid signature")
)

/*
Notary is the central actor for custody and economics in Caramba.
It maintains the Ledger, verifies identities, and processes transactions
when compute jobs are successfully completed across the network.
*/
type Notary struct {
	identity *Identity
	ledger   *Ledger
}

/*
NewNotary creates a new Notary instance with an empty ledger.
*/
func NewNotary() *Notary {
	identity, err := NewIdentity()

	if err != nil {
		panic(fmt.Errorf("notary: failed to create mint authority: %w", err))
	}

	return &Notary{
		identity: identity,
		ledger:   NewLedger(identity.Address()),
	}
}

/*
Ledger returns a reference to the Notary's underlying ledger.
*/
func (n *Notary) Ledger() *Ledger {
	return n.ledger
}

/*
MintCredits signs and records a controlled credit issuance.
*/
func (n *Notary) MintCredits(recipientAddress string, amount int64) error {
	nonce := n.ledger.MintNonce() + 1
	payload := MintPayload(n.identity.Address(), recipientAddress, amount, nonce)
	signature := n.identity.Sign(payload)

	return n.ledger.Mint(n.identity.Address(), recipientAddress, amount, signature)
}

/*
SubmitManifest is called when a user wants to submit a research intent to the network.
The Notary verifies the user's signature, deducts the estimated credit cost,
and records the manifest's hash in the ledger as proof of intent.
*/
func (n *Notary) SubmitManifest(sender *Identity, manifestData []byte, signature []byte, estimatedCost int64) (string, error) {
	senderAddr := sender.Address()

	if !Verify(senderAddr, manifestData, signature) {
		return "", ErrInvalidSignature
	}

	if n.ledger.BalanceOf(senderAddr) < estimatedCost {
		return "", ErrInsufficientFunds
	}

	// Escrow the funds: transfer to a "burn" or "escrow" address.
	// For simplicity, we just deduct them from the sender.
	if err := n.ledger.Transfer(senderAddr, "escrow", estimatedCost); err != nil {
		return "", err
	}

	hashBytes := sha256.Sum256(manifestData)
	manifestHash := hex.EncodeToString(hashBytes[:])

	n.ledger.RecordArtifact(manifestHash, senderAddr)

	return manifestHash, nil
}

/*
SettleCompute resolves a distributed compute job. When a volunteer (worker)
successfully processes an IR graph for a requester (owner), the Notary
verifies the worker's proof of work (signature), and transfers credits
from the escrow to the worker.
*/
func (n *Notary) SettleCompute(ownerAddr string, worker *Identity, resultData []byte, workerSignature []byte, payout int64) error {
	workerAddr := worker.Address()

	if !Verify(workerAddr, resultData, workerSignature) {
		return ErrInvalidSignature
	}

	// Transfer funds from escrow to the worker who completed the job.
	if err := n.ledger.Transfer("escrow", workerAddr, payout); err != nil {
		return fmt.Errorf("notary: failed to settle payout: %w", err)
	}

	// Optionally record the artifact hash of the result to the owner's provenance.
	resultHashBytes := sha256.Sum256(resultData)
	resultHash := hex.EncodeToString(resultHashBytes[:])
	n.ledger.RecordArtifact(resultHash, ownerAddr)

	return nil
}
