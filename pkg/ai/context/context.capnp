@0xd4c9c9f76e88a0d6;

using Go = import "/go.capnp";
$Go.package("context");
$Go.import("github.com/theapemachine/caramba/pkg/ai/context");

struct Context {
    messages @0 :List(Message);
}

struct Message {
    id        @0 :Text;
    role      @1 :Text;
    name      @2 :Text;
    content   @3 :Text;
    toolCalls @4 :List(ToolCall);
}

struct ToolCall {
    id        @0 :Text;
    name      @1 :Text;
    arguments @2 :Text;
}

struct Prompt {
    fragments @0 :List(Fragment);
}

struct Fragment {
    template  @0 :Text;
    variables @1 :List(Text);
}