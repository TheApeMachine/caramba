@0xd4c9c9f76e88a0d0;

using Go = import "/go.capnp";
$Go.package("agent");
$Go.import("github.com/theapemachine/caramba/pkg/ai/agent");

using Params  = import "../params/params.capnp";
using Context = import "../context/context.capnp";
using Tools   = import "../../tools/tool.capnp";

struct Agent {
    identity @0 :Identity;
    params   @1 :Params.Params;
    context  @2 :Context.Context;
    tools    @3 :List(Tools.Tool);
}

struct Identity {
    identifier @0 :Text;
    name       @1 :Text;
    role       @2 :Text;
}

interface State {
    get @0 () -> (params :Params.Params, context :Context.Context);
    set @1 (params :Params.Params, context :Context.Context);
}