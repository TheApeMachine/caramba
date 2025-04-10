@0xd4c9c9f76e88a0d0;

using Go = import "/go.capnp";
$Go.package("agent");
$Go.import("github.com/theapemachine/caramba/pkg/ai/agent");

using Params   = import "../params/params.capnp";
using Context  = import "../context/context.capnp";
using Artifact = import "../../datura/artifact.capnp";
using Message  = import "../message/message.capnp";
using Tool     = import "../tool/tool.capnp";

struct Agent {
    identity @0 :Identity;
    state    @1 :UInt64;
    params   @2 :Params.Params;
    context  @3 :Context.Context;
    tools    @4 :List(Tool.Tool);
}

struct Identity {
    identifier @0 :Text;
    name       @1 :Text;
    role       @2 :Text;
}

interface RPC {
    send @0 (artifact :Artifact.Artifact) -> (out :Artifact.Artifact);
}