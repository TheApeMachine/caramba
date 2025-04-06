# Datura 🌸

Datura is a powerful Go package for secure artifact management with built-in encryption, metadata handling, and zero-knowledge proofs. It provides a robust foundation for creating, managing, and securing data artifacts with cryptographic guarantees.

## Why Datura? 🤔

Datura was created to solve a fundamental challenge in systems built around Go's `io.Reader` and `io.Writer` interfaces: how to maintain rich data context while working with byte streams. In a system where everything flows through these interfaces, traditional approaches often require constant type conversion and context switching, leading to performance overhead and complexity.

Datura solves this by providing a single, flexible `Artifact` type that:

- Implements `io.Reader` and `io.Writer` for seamless integration with streaming operations
- Preserves data context through metadata, roles, and scopes
- Maintains high performance using Cap'n Proto serialization
- Adds security and privacy features without compromising the simplicity of the streaming interface

This means you can work with any type of data - from simple text to complex structures - while maintaining context about what the data represents, how it should be handled, and who should access it, all through a consistent streaming interface.

## Features ✨

- **Secure Artifact Management**: Create and manage artifacts with automatic UUID generation and timestamping
- **End-to-End Encryption**: Built-in AES-GCM encryption with ephemeral key pairs for secure payload handling
- **Flexible Metadata**: Support for multiple data types (text, integers, floats, booleans, binary) in metadata
- **Zero-Knowledge Proofs**: Integration with zk-SNARKs for privacy-preserving authentication
- **Role-Based Access**: Built-in role system (System, User, Assistant) for access control
- **Scope Management**: Support for different artifact scopes (Event, Message, Prompt)
- **Media Type Handling**: Comprehensive support for various media types
- **IO Interface Implementation**: Standard Go io.Reader and io.Writer interface support
- **Cap'n Proto Integration**: High-performance serialization using Cap'n Proto

## Installation 📦

```bash
go get github.com/theapemachine/caramba/pkg/datura
```

## Quick Start 🚀

```go
import "github.com/theapemachine/caramba/pkg/datura"

// Create a new artifact with payload
artifact := datura.New(
    datura.WithMediatype(datura.MediaTypeTextPlain),
    datura.WithRole(datura.ArtifactRoleUser),
    datura.WithScope(datura.ArtifactScopePrompt),
    datura.WithEncryptedPayload([]byte("Hello, World!")),
)

// Add metadata
metadata := map[string]any{
    "version": "1.0",
    "author": "John Doe",
}
datura.WithMetadata(metadata)(artifact)

// Decrypt payload
decryptedPayload, err := artifact.DecryptPayload()
if err != nil {
    // Handle error
}
```

## Core Components 🛠️

### Artifact

The `Artifact` type is the central component of Datura, representing a secure data container with the following features:

- Unique UUID identification
- Timestamp tracking
- Encrypted payload management
- Metadata storage
- Role and scope assignment
- Zero-knowledge proof support

### CryptoSuite 🔐

The `CryptoSuite` handles all cryptographic operations:

- AES-GCM encryption/decryption
- Ephemeral key pair generation
- Secure key management

### Metadata 📝

Flexible metadata system supporting multiple value types:

- Text strings
- Integers (64-bit)
- Floating-point numbers
- Boolean values
- Binary data

### Roles and Scopes 🎭

Pre-defined roles:

- `ArtifactRoleSystem`
- `ArtifactRoleUser`
- `ArtifactRoleAssistant`
- `ArtifactRoleOpenFile`
- `ArtifactRoleSaveFile`
- And more...

Available scopes:

- `ArtifactScopeEvent`
- `ArtifactScopeMessage`
- `ArtifactScopePrompt`

## Advanced Usage 💡

### Working with Zero-Knowledge Proofs

```go
// Generate a proof
circuit := &datura.AuthCircuit{
    PseudonymHash: pseudonymHash,
    MerkleRoot: merkleRoot,
}
proof, err := datura.GenerateProof(artifact, provingKey)
```

### Custom Metadata Handling

```go
// Get typed metadata
value := datura.GetMetaValue[string](artifact, "key")
intValue := datura.GetMetaValue[int](artifact, "number")
```

## Performance ⚡

Datura uses Cap'n Proto for high-performance serialization, making it suitable for high-throughput applications. The encryption and decryption operations are optimized for efficiency while maintaining strong security guarantees.
