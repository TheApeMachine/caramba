@0xd4c9c9f76e88a0d4;

using Go = import "/go.capnp";
$Go.package("core");
$Go.import("github.com/theapemachine/caramba/pkg/core");

struct Params {
    model            @0 :Text;
    temperature      @1 :Float64;
    topP             @2 :Float64;
    topK             @3 :Float64;
    frequencyPenalty @4 :Float64;
    presencePenalty  @5 :Float64;
    maxTokens        @6 :Float64;
    stream           @7 :Bool;
}

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