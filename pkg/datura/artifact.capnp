using Go = import "/go.capnp";
@0x85d3acc39d94e0f8;
$Go.package("datura");
$Go.import("datura/artifact");

struct Artifact {
    uuid @0 :Data;
    checksum @1 :Data;
    timestamp @2 :Int64;
    mediatype @3 :Text;
    role @4 :UInt32;
    scope @5 :UInt32;
    pseudonymHash @6 :Data;  # zk-SNARK identity hash
    merkleRoot @7 :Data;  # Root of the Merkle Tree

    struct Metadata {
        key @0 :Text;
        value :union {
            textValue @1 :Text;
            intValue @2 :Int64;
            floatValue @3 :Float64;
            boolValue @4 :Bool;
            binaryValue @5 :Data;
        }
    }
    metadata @8 :List(Metadata);

    encryptedPayload @9 :Data;
    encryptedKey @10 :Data;
    ephemeralPublicKey @11 :Data;

    struct Approval {
        zkProof @0 :Data;  # Employee's zero-knowledge proof
        operatorBlindSignature @1 :Data;  # Operator's blind signature approval
    }
    approvals @12 :List(Approval);

    signature @13 :Data;
}