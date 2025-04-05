@0xd4c9c9f76e88a0d2;

using Go = import "/go.capnp";
$Go.package("provider");
$Go.import("github.com/theapemachine/caramba/pkg/ai/provider");

using Params  = import "../params/params.capnp";
using Context = import "../context/context.capnp";
using Tools   = import "../tool/tool.capnp";

struct Provider {
    name @0 :Text;
}

interface Generate {
    call   @0 (params :Params.Params, context :Context.Context, tools :List(Tools.Tool)) -> (out :Text);
}

interface ByteStream {
    write @0 (data :Data) -> stream;
    done  @1 ();
}
