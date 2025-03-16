using Go = import "/go.capnp";
@0xe363a5839bf866c6;
$Go.package("context");
$Go.import("context/artifact");

struct Artifact {
    id @0 :Text;
    model @1 :Text;
    messages @2 :List(Message);
    tools @3 :List(Tool);
    process @4 :Data;
    temperature @5 :Float64;
    topP @6 :Float64;
    topK @7 :Float64;
    presencePenalty @8 :Float64;
    frequencyPenalty @9 :Float64;
    maxTokens @10 :Int32;
    stopSequences @11 :List(Text);
    stream @12 :Bool;
}

struct Message {
    id @0 :Text;
    role @1 :Text;
    name @2 :Text;
    content @3 :Text;
}

struct Tool {
    id @0 :Text;
    name @1 :Text;
    description @2 :Text;
    parameters @3 :List(Parameter);
}

struct Parameter {
    type @0 :Text;
    properties @1 :List(Property);
    required @2 :Bool;
    additionalProperties @3 :Bool;
}

struct Property {
    type @0 :Text;
    description @1 :Text;
    enum @2 :List(Text);
}
