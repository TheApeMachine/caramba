@0xd4c9c9f76e88a0d1;

using Go = import "/go.capnp";
$Go.package("tools");
$Go.import("github.com/theapemachine/caramba/pkg/tools");

struct Tool {
  function @0 :Function;
}

struct Function {
  name        @0 :Text;
  description @1 :Text;
  parameters  @2 :Parameters;
}

struct Parameters {
  type       @0 :Text = "object";
  properties @1 :List(Property);
  required   @2 :List(Text);
}

struct Property {
  name        @0 :Text;
  type        @1 :Text;
  description @2 :Text;
  enum        @3 :List(Text);
}
