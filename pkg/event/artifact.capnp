using Go = import "/go.capnp";
@0xe363a5839bf866c4;
$Go.package("event");
$Go.import("event/artifact");

struct Artifact {
    id @0 :Text;
    type @1 :Text;
    timestamp @2 :Int64;
    origin @3 :Text;
    role @4 :Text;
    payload @5 :Data;
}

struct Map(Key, Value) {
  entries @0 :List(Entry);
  struct Entry {
    key @0 :Key;
    value @1 :Value;
  }
}
