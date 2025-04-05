using Go = import "/go.capnp";
@0x85d3acc39d94e0f8;
$Go.package("datura");
$Go.import("github.com/theapemachine/caramba/pkg/datura");

struct Artifact {
    uuid          @0 :Data;
    checksum      @1 :Data;
    timestamp     @2 :Int64;
    mediatype     @3 :Text;
    origin        @4 :Text;
    issuer        @5 :Text;
    role          @6 :UInt32;
    scope         @7 :UInt32;
    pseudonymHash @8 :Data;
    merkleRoot    @9 :Data;
    struct Metadata {
        key @0 :Text;
        value :union {
            textValue   @1 :Text;
            intValue    @2 :Int64;
            floatValue  @3 :Float64;
            boolValue   @4 :Bool;
            binaryValue @5 :Data;
        }
    }
    metadata           @10 :List(Metadata);
    encryptedPayload   @11 :Data;
    encryptedKey       @12 :Data;
    ephemeralPublicKey @13 :Data;
    struct Approval {
        zkProof                @0 :Data;
        operatorBlindSignature @1 :Data;
    }
    approvals @14 :List(Approval);
    signature @15 :Data;
}

