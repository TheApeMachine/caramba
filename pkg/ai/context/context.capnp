@0xd4c9c9f76e88a0d6;

using Go = import "/go.capnp";
$Go.package("context");
$Go.import("github.com/theapemachine/caramba/pkg/ai/context");

using Message = import "../message/message.capnp";

struct Context {
    uuid     @0 :Text;
    state    @1 :UInt64;
    messages @2 :List(Message.Message);
}

interface RPC {
    add @0 (context :Message.Message);
}