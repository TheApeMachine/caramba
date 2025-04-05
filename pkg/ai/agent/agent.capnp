@0xd4c9c9f76e88a0d0;

using Go = import "/go.capnp";
$Go.package("agent");
$Go.import("github.com/theapemachine/caramba/pkg/ai/agent");

using Params   = import "../params/params.capnp";
using Context  = import "../context/context.capnp";
using Artifact = import "../../datura/artifact.capnp";

struct Agent {
    identity @0 :Identity;
    params   @1 :Params.Params;
    context  @2 :Context.Context;
    tools    @3 :List(Text);
}

struct Identity {
    identifier @0 :Text;
    name       @1 :Text;
    role       @2 :Text;
}

interface AgentRPC {
    run     @0 () -> stream;
    stop    @1 () -> ();    
}