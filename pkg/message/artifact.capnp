using Go = import "/go.capnp";
@0xe363a5839bf866c5;
$Go.package("message");
$Go.import("message/artifact");

struct Artifact {
  id @0 :Text;
  role @1 :Text;
  name @2 :Text;
  content @3 :Text;
}
