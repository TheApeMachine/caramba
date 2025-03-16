using Go = import "/go.capnp";
@0xe363a5839bf866c9;
$Go.package("tool");
$Go.import("tool/artifact");

struct Artifact {
    type @0 :Text;
    function @1 :Function;
}

struct Function {
    name @0 :Text;
    description @1 :Text;
    parameters @2 :List(Parameter);
    strict @3 :Bool;
}

struct Parameter {
    type @0 :Text;
    properties @1 :List(Property);
    required @2 :List(Text);
    additionalProperties @3 :Bool;
}

struct Property {
    type @0 :Text;
    name @1 :Text;
    description @2 :Text;
    enum @3 :List(Text);
}