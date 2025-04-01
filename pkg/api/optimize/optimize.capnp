@0xefed8c3251756e7a;  # Unique file ID

using Go = import "/go.capnp";
$Go.package("optimize");
$Go.import("github.com/theapemachine/caramba/pkg/api/optimize");

struct Optimize {
    operation @0 :Text;
    system @1 :Text;
    temperature @2 :Float64;
    topP @3 :Float64;
    topK @4 :Float64;
    frequencyPenalty @5 :Float64;
    presencePenalty @6 :Float64;
}
