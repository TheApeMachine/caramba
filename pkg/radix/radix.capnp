@0xd3c5d01f75666abd;  # Your unique ID here

using Go = import "/go.capnp";
$Go.package("radix");
$Go.import("github.com/theapemachine/caramba/pkg/radix");

using Artifact = import "../datura/artifact.capnp";

# Message types for radix tree network communication
struct RadixMessage {
  type @0 :MessageType;
  union {
    insert @1 :InsertPayload;
    sync @2 :SyncPayload;
    proof @3 :ProofPayload;
    requestVote @4 :RequestVotePayload;
    heartbeat @5 :HeartbeatPayload;
  }
}

enum MessageType {
  insert @0;
  sync @1;
  proof @2;
  requestVote @3;
  heartbeat @4;
}

struct InsertPayload {
  key @0 :Data;
  artifact @1 :Artifact.Artifact;
  merkleProof @2 :List(Data);  # Add Merkle proof
}

struct SyncPayload {
  merkleRoot @0 :Data;
  entries @1 :List(SyncEntry);
  proofs @2 :List(MerkleProof);  # Add list of proofs
}

struct SyncEntry {
  key @0 :Data;
  artifact @1 :Artifact.Artifact;
}

struct ProofPayload {
  key @0 :Data;
  value @1 :Data;
  proof @2 :List(Data);
}

struct MerkleProof {
  key @0 :Data;
  proof @1 :List(Data);
}

struct RecoverPayload {
  lastKnownMerkleRoot @0 :Data;
}

struct RequestVotePayload {
  term @0 :UInt64;
  candidateId @1 :Text;
  lastLogIndex @2 :UInt64;
}

struct HeartbeatPayload {
  term @0 :UInt64;
  leaderId @1 :Text;
}

interface RadixRPC {
  # Insert an artifact into the tree
  insert @0 (key :Data, artifact :Artifact.Artifact) -> (success :Bool);
  
  # Request synchronization with peers
  sync @1 (merkleRoot :Data) -> (diff :SyncPayload);
  
  # Request recovery after restart
  recover @2 (lastKnownMerkleRoot :Data) -> (state :SyncPayload);

  # Request vote for leader election
  requestVote @3 (term :UInt64, candidateId :Text, lastLogIndex :UInt64) -> (term :UInt64, voteGranted :Bool);

  # Send heartbeat to maintain leadership
  heartbeat @4 (term :UInt64, leaderId :Text) -> (term :UInt64, success :Bool);
} 