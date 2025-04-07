@0xd4c9c9f76e88a0d1;

using Go = import "/go.capnp";
$Go.package("tool");
$Go.import("github.com/theapemachine/caramba/pkg/ai/tool");

using Artifact = import "../../datura/artifact.capnp";

struct Tool {
    name       @0 :Text;
    operations @1 :List(Operation);
}

struct Operation {
    name        @0 :Text;
    description @1 :Text;
    parameters  @2 :List(Parameter);
    required    @3 :List(Text);
}

struct Parameter {
    name        @0 :Text;
    type        @1 :Text;       # Represents the data type (e.g., "string", "number", "boolean")
    description @2 :Text;       # Description of the parameter
    enum        @3 :List(Text); # Optional list of allowed enum values
}

interface RPC {
    use @0 (artifact :Artifact.Artifact) -> (out :Artifact.Artifact);
}