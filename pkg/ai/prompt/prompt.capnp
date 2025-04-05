@0xd4c9c9f76e88a0e1;

using Go = import "/go.capnp";
$Go.package("prompt");
$Go.import("github.com/theapemachine/caramba/pkg/ai/prompt");

struct Prompt {
    fragments @0 :List(Fragment);
}

struct Fragment {
    template  @0 :Text;
    variables @1 :List(Text);
}