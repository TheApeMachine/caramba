@0xd4c9c9f76e88a0d2;

using Go = import "/go.capnp";
$Go.package("provider");
$Go.import("github.com/theapemachine/caramba/pkg/ai/provider");

using Artifact = import "../../datura/artifact.capnp";

struct Provider {
    name @0 :Text;
}

interface RPC {
    generate @0 (artifact :Artifact.Artifact) -> (out :Artifact.Artifact);
}
