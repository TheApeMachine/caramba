package provenance

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"sync"
	"time"
)

/*
Ledger is the per-run provenance record. The runtime accumulates
entries as a run progresses (program hashes, asset hashes, seeds,
state snapshots, output artifacts, telemetry events). Serialize
emits a canonical JSON document; Digest returns SHA-256 of that
document. Sign produces an Ed25519 signature over the digest so
researchers can attach a key and prove a run produced a specific
artifact set.
*/
type Ledger struct {
	mu       sync.Mutex
	created  time.Time
	entries  []Entry
	metadata map[string]any
}

/*
Entry is one event in the ledger. Kind discriminates the payload.
The runtime serializes entries in insertion order so digests are
stable across replays of the same program.
*/
type Entry struct {
	Kind    string         `json:"kind"`
	Name    string         `json:"name"`
	Hash    string         `json:"hash,omitempty"`
	Path    string         `json:"path,omitempty"`
	Seed    int64          `json:"seed,omitempty"`
	Payload map[string]any `json:"payload,omitempty"`
	Time    time.Time      `json:"time"`
}

/*
New constructs an empty Ledger with the supplied opaque metadata
preserved in the serialized form.
*/
func New(metadata map[string]any) *Ledger {
	copied := map[string]any{}

	for key, value := range metadata {
		copied[key] = value
	}

	return &Ledger{
		created:  time.Now().UTC(),
		metadata: copied,
	}
}

/*
RecordProgram captures a program's name + SHA-256 hash of the
serialized IR (or any other byte-stable representation).
*/
func (ledger *Ledger) RecordProgram(name string, programBytes []byte) {
	ledger.append(Entry{
		Kind: "program",
		Name: name,
		Hash: sha256Hex(programBytes),
	})
}

/*
RecordAsset captures an asset id + path + content hash. Used for
model weights, tokenizer files, datasets.
*/
func (ledger *Ledger) RecordAsset(name string, path string, contentHash string) {
	ledger.append(Entry{
		Kind: "asset",
		Name: name,
		Path: path,
		Hash: contentHash,
	})
}

/*
RecordSeed captures a named RNG seed.
*/
func (ledger *Ledger) RecordSeed(name string, seed int64) {
	ledger.append(Entry{
		Kind: "seed",
		Name: name,
		Seed: seed,
	})
}

/*
RecordOutput captures an output artifact's path and content hash.
*/
func (ledger *Ledger) RecordOutput(name string, path string, contentHash string) {
	ledger.append(Entry{
		Kind: "output",
		Name: name,
		Path: path,
		Hash: contentHash,
	})
}

/*
RecordSnapshot captures a state-object snapshot with its hash.
*/
func (ledger *Ledger) RecordSnapshot(name string, payload []byte) {
	ledger.append(Entry{
		Kind: "snapshot",
		Name: name,
		Hash: sha256Hex(payload),
		Payload: map[string]any{
			"bytes": len(payload),
		},
	})
}

/*
RecordEvent attaches an arbitrary named event with structured
fields. Used for telemetry-derived ledger entries.
*/
func (ledger *Ledger) RecordEvent(name string, fields map[string]any) {
	copied := map[string]any{}

	for key, value := range fields {
		copied[key] = value
	}

	ledger.append(Entry{
		Kind:    "event",
		Name:    name,
		Payload: copied,
	})
}

func (ledger *Ledger) append(entry Entry) {
	entry.Time = time.Now().UTC()

	ledger.mu.Lock()
	defer ledger.mu.Unlock()

	ledger.entries = append(ledger.entries, entry)
}

/*
Entries returns a copy of the recorded entries in insertion order.
*/
func (ledger *Ledger) Entries() []Entry {
	ledger.mu.Lock()
	defer ledger.mu.Unlock()

	out := make([]Entry, len(ledger.entries))
	copy(out, ledger.entries)

	return out
}

/*
Serialize emits the ledger as canonical JSON. Metadata keys are
sorted so the byte-form is reproducible for digesting.
*/
func (ledger *Ledger) Serialize() ([]byte, error) {
	ledger.mu.Lock()
	defer ledger.mu.Unlock()

	document := map[string]any{
		"created":  ledger.created.Format(time.RFC3339Nano),
		"metadata": sortedMap(ledger.metadata),
		"entries":  ledger.entries,
	}

	return json.MarshalIndent(document, "", "  ")
}

/*
Digest returns SHA-256 of Serialize. Errors propagate from
Serialize so callers can decide how to handle them.
*/
func (ledger *Ledger) Digest() ([]byte, error) {
	bytes, err := ledger.Serialize()

	if err != nil {
		return nil, err
	}

	sum := sha256.Sum256(bytes)

	return sum[:], nil
}

/*
Sign signs the digest with the supplied Ed25519 private key. The
result is the raw signature bytes; callers attach them alongside
the ledger so a verifier can reproduce the digest and check the
signature.
*/
func (ledger *Ledger) Sign(privateKey ed25519.PrivateKey) ([]byte, error) {
	digest, err := ledger.Digest()

	if err != nil {
		return nil, err
	}

	return ed25519.Sign(privateKey, digest), nil
}

/*
Verify checks an existing signature against the ledger's digest
using the supplied public key.
*/
func (ledger *Ledger) Verify(publicKey ed25519.PublicKey, signature []byte) (bool, error) {
	digest, err := ledger.Digest()

	if err != nil {
		return false, err
	}

	return ed25519.Verify(publicKey, digest, signature), nil
}

/*
WriteFile serializes the ledger to disk at path. The file is
overwritten if it exists.
*/
func (ledger *Ledger) WriteFile(path string) error {
	bytes, err := ledger.Serialize()

	if err != nil {
		return err
	}

	return os.WriteFile(path, bytes, 0o644)
}

/*
WriteTo emits the serialized ledger to the supplied writer.
*/
func (ledger *Ledger) WriteTo(writer io.Writer) (int64, error) {
	bytes, err := ledger.Serialize()

	if err != nil {
		return 0, err
	}

	count, err := writer.Write(bytes)

	return int64(count), err
}

func sha256Hex(payload []byte) string {
	sum := sha256.Sum256(payload)

	return hex.EncodeToString(sum[:])
}

func sortedMap(source map[string]any) map[string]any {
	keys := make([]string, 0, len(source))

	for key := range source {
		keys = append(keys, key)
	}

	sort.Strings(keys)

	out := map[string]any{}

	for _, key := range keys {
		out[key] = source[key]
	}

	return out
}

/*
ReadFile loads a serialized ledger from disk back into memory. The
created timestamp and metadata are recovered; entries arrive in the
order they were originally written so digest is reproducible.
*/
func ReadFile(path string) (*Ledger, error) {
	bytes, err := os.ReadFile(path)

	if err != nil {
		return nil, fmt.Errorf("provenance: read %s: %w", path, err)
	}

	return decode(bytes)
}

func decode(payload []byte) (*Ledger, error) {
	var document struct {
		Created  string         `json:"created"`
		Metadata map[string]any `json:"metadata"`
		Entries  []Entry        `json:"entries"`
	}

	if err := json.Unmarshal(payload, &document); err != nil {
		return nil, fmt.Errorf("provenance: decode: %w", err)
	}

	parsedTime, err := time.Parse(time.RFC3339Nano, document.Created)

	if err != nil {
		return nil, fmt.Errorf("provenance: parse created: %w", err)
	}

	ledger := &Ledger{
		created:  parsedTime.UTC(),
		metadata: document.Metadata,
		entries:  document.Entries,
	}

	return ledger, nil
}
