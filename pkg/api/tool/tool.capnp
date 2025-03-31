@0xd4c9c9f76e89a0d7;  # Unique file ID

using Go = import "/go.capnp";
$Go.package("tool");
$Go.import("github.com/theapemachine/caramba/pkg/api/tool");

struct Tool {
  function @0 :Function;
}

struct Function {
  name @0 :Text;
  description @1 :Text;
  parameters @2 :Parameters;
}

struct Parameters {
  type @0 :Text = "object";
  properties @1 :List(Property);
  required @2 :List(Text);
}

struct Property {
  name @0 :Text;
  type @1 :Text;
  description @2 :Text;
  enum @3 :List(Text);
}
