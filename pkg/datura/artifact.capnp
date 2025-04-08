using Go = import "/go.capnp";
@0x85d3acc39d94e0f8;
$Go.package("datura");
$Go.import("github.com/theapemachine/caramba/pkg/datura");

struct Artifact {
    uuid          @0  :Text;
    state         @1  :UInt64;
    checksum      @2  :Data;
    timestamp     @3  :Int64;
    mediatype     @4  :Text;
    origin        @5  :Text;
    issuer        @6  :Text;
    role          @7  :UInt32;
    scope         @8  :UInt32;
    pseudonymHash @9  :Data;
    merkleRoot    @10 :Data;
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
    metadata           @11 :List(Metadata);
    payload            @12 :Data;
    encryptedPayload   @13 :Data;
    encryptedKey       @14 :Data;
    ephemeralPublicKey @15 :Data;
    struct Approval {
        zkProof                @0 :Data;
        operatorBlindSignature @1 :Data;
    }
    approvals @16 :List(Approval);
    signature @17 :Data;
}

