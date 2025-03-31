@0xefed8c3251756e9a;  # Unique file ID

using Go = import "/go.capnp";
$Go.package("system");
$Go.import("github.com/theapemachine/caramba/pkg/api/system");

struct Agent {
    name @0 :Text;
}

struct System {
  agents @0 :List(Agent);
}
