@0xd4c9c9f76e88a0d0;

using Go = import "/go.capnp";
$Go.package("ai");
$Go.import("github.com/theapemachine/caramba/pkg/ai");

# Import common types
using Tools = import "../tools/tool.capnp";

struct Agent {
    identity @0 :Identity;
    tools    @1 :List(Tools.Tool);
}

struct Identity {
    identifier @0 :Text;
    name       @1 :Text;
    role       @2 :Text;
}
