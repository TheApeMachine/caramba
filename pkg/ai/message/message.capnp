@0xd4c9c9f76e88a0d9;

using Go = import "/go.capnp";
$Go.package("message");
$Go.import("github.com/theapemachine/caramba/pkg/ai/message");

using ToolCall = import "../toolcall/toolcall.capnp";

struct Message {
    uuid      @0 :Text;
    state     @1 :UInt64;
    id        @2 :Text;
    role      @3 :Text;
    name      @4 :Text;
    content   @5 :Text;
    toolCalls @6 :List(ToolCall.ToolCall);
}
