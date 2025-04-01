@0xd4c9c9f76e88a0d2;

using Go = import "/go.capnp";
$Go.package("provider");
$Go.import("github.com/theapemachine/caramba/pkg/provider");

struct Provider {
    name @0 :Text;
}
