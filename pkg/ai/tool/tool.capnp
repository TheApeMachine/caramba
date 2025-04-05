@0xd4c9c9f76e88a0d1;

using Go = import "/go.capnp";
$Go.package("tool");
$Go.import("github.com/theapemachine/caramba/pkg/ai/tool");

struct Tool {
    name @0 :Text;
}
