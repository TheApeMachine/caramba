@0xd4c9c9f76e88a0d2;

using Go = import "/go.capnp";
$Go.package("provider");
$Go.import("github.com/theapemachine/caramba/pkg/ai/provider");

using Artifact = import "../../datura/artifact.capnp";

struct Provider {
    uuid  @0 :Text;
    state @1 :UInt64;
    name  @2 :Text;
}

interface RPC {
    generate @0 (artifact :Artifact.Artifact) -> (out :Artifact.Artifact);
}
