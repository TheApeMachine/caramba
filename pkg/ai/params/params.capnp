@0xd4c9c9f76e88a0d4;

using Go = import "/go.capnp";
$Go.package("params");
$Go.import("github.com/theapemachine/caramba/pkg/ai/params");

struct Params {
    model            @0 :Text;
    temperature      @1 :Float64;
    topP             @2 :Float64;
    topK             @3 :Float64;
    frequencyPenalty @4 :Float64;
    presencePenalty  @5 :Float64;
    maxTokens        @6 :Float64;
    stream           @7 :Bool;
    format           @8 :ResponseFormat;
}

struct ResponseFormat {
    name        @0 :Text;
    description @1 :Text;
    schema      @2 :Text;
    strict      @3 :Bool;
}
