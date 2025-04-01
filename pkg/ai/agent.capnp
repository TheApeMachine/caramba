@0xd4c9c9f76e88a0d0;

using Go = import "/go.capnp";
$Go.package("ai");
$Go.import("github.com/theapemachine/caramba/pkg/ai");

# Import common types
using Provider = import "../provider/provider.capnp";
using Tools = import "../tools/tool.capnp";

struct Agent {
    identity @0 :Identity;
    provider @1 :Provider.Provider;
    params   @2 :Params;
    context  @3 :Context;
    tools    @4 :List(Tools.Tool);
}

struct Identity {
    identifier @0 :Text;
    name       @1 :Text;
    role       @2 :Text;
}

struct Params {
    model       @0 :Text;
    temperature @1 :Float64;
}

struct Context {
    messages @0 :List(Message);
}

struct Message {
    id      @0 :Text;
    role    @1 :Text;
    name    @2 :Text;
    content @3 :Text;
}
