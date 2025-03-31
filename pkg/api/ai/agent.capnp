@0xd4c9c9f76e89a0d3;  # Unique file ID

using Go = import "/go.capnp";
$Go.package("ai");
$Go.import("github.com/theapemachine/caramba/pkg/api/ai");

# Import common types
using Import = import "../provider/provider.capnp";

struct Agent {
    name @0 :Text;
    context @1 :Import.ProviderParams;
}
