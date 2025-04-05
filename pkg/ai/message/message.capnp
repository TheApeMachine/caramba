@0xd4c9c9f76e88a0d9;

using Go = import "/go.capnp";
$Go.package("message");
$Go.import("github.com/theapemachine/caramba/pkg/ai/message");

using ToolCall = import "../toolcall/toolcall.capnp";

struct Message {
    id        @0 :Text;
    role      @1 :Text;
    name      @2 :Text;
    content   @3 :Text;
    toolCalls @4 :List(ToolCall.ToolCall);
}
