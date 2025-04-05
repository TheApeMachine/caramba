@0xd4c9c9f76e88a0e0;

using Go = import "/go.capnp";
$Go.package("toolcall");
$Go.import("github.com/theapemachine/caramba/pkg/ai/toolcall");

struct ToolCall {
    id        @0 :Text;
    name      @1 :Text;
    arguments @2 :Text;
}
