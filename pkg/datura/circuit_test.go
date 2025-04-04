package datura

import (
	"bytes"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"capnproto.org/go/capnp/v3"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

func setupTestArtifact() (Artifact, error) {
	_, seg, err := capnp.NewMessage(capnp.SingleSegment(nil))
	if err != nil {
		return Artifact{}, err
	}

	mockArtifact, err := NewArtifact(seg)
	if err != nil {
		return Artifact{}, err
	}

	// Mock pseudonym hash and Merkle root (fake test values)
	pseudonymHash := []byte{1, 2, 3, 4}
	merkleRoot := []byte{9, 8, 7, 6}

	if err := mockArtifact.SetPseudonymHash(pseudonymHash); err != nil {
		return Artifact{}, err
	}
	if err := mockArtifact.SetMerkleRoot(merkleRoot); err != nil {
		return Artifact{}, err
	}

	return mockArtifact, nil
}

func setupCircuitKeys() (groth16.ProvingKey, groth16.VerifyingKey, error) {
	var circuit AuthCircuit
	modulus := fr.Modulus()
	r1cs, err := frontend.Compile(modulus, r1cs.NewBuilder, &circuit)
	if err != nil {
		return nil, nil, err
	}

	return groth16.Setup(r1cs)
}

func TestGenerateProof(t *testing.T) {
	Convey("Given an artifact, should successfully generate a zk-SNARK proof", t, func() {
		mockArtifact, err := setupTestArtifact()
		So(err, ShouldBeNil)

		pk, _, err := setupCircuitKeys()
		So(err, ShouldBeNil)

		proof, err := GenerateProof(mockArtifact, pk)
		So(err, ShouldBeNil)
		So(proof, ShouldNotBeNil)

		// Verify proof can be deserialized
		proofStruct := groth16.NewProof(bn254.ID)
		buf := bytes.NewBuffer(proof)
		_, err = proofStruct.ReadFrom(buf)
		So(err, ShouldBeNil)
	})
}

func TestVerifyProof(t *testing.T) {
	Convey("Given a valid proof, should successfully verify it", t, func() {
		mockArtifact, err := setupTestArtifact()
		So(err, ShouldBeNil)

		pk, vk, err := setupCircuitKeys()
		So(err, ShouldBeNil)

		proof, err := GenerateProof(mockArtifact, pk)
		So(err, ShouldBeNil)

		valid := VerifyProof(mockArtifact, proof, vk)
		So(valid, ShouldBeTrue)
	})
}
