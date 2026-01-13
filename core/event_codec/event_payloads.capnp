@0xa1b2c3d4e5f60002;

# Typed payloads for EventEnvelope.payload (Cap'n Proto bytes).
#
# NOTE: These structs are serialized independently and stored as opaque bytes
# in EventEnvelope.payload. The EventEnvelope itself remains a stable wrapper
# that carries routing metadata (type, sender, priority, etc.).

struct Metric {
  key @0 :Text;
  value @1 :Float64;
}

struct MessagePayload {
  text @0 :Text;
}

struct MemoryWritePayload {
  key @0 :UInt32;
  value @1 :UInt32;
}

struct MemoryQueryPayload {
  key @0 :UInt32;
}

struct MemoryAnswerPayload {
  value @0 :UInt32;
}

struct NoisePayload {
  tok @0 :UInt32;
}

struct IdlePayload {
  ts @0 :Float64;
  metrics @1 :List(Metric);
}

struct ToolBuilderDefinitionPayload {
  name @0 :Text;
  description @1 :Text;
  implementation @2 :Text;
  requirements @3 :List(Text);
}

struct ToolBuilderRegisteredPayload {
  name @0 :Text;
  path @1 :Text;
}

struct ToolBuilderRejectedPayload {
  error @0 :Text;
}

struct DriveBandPayload {
  minValue @0 :Float64;
  maxValue @1 :Float64;
}

struct DriveSignalPayload {
  name @0 :Text;
  metric @1 :Text;
  value @2 :Float64;
  band @3 :DriveBandPayload;
  deviation @4 :Float64;
  urgency @5 :Float64;
}

struct ImpulsePayload {
  metrics @0 :List(Metric);
  signals @1 :List(DriveSignalPayload);
  maxUrgency @2 :Float64;
}

struct ToolchainCapabilitiesPayload {
  filesystem @0 :Bool;
  network @1 :Bool;
  process @2 :Bool;
  clock @3 :Bool;
}

struct ToolchainDefinitionPayload {
  name @0 :Text;
  version @1 :Text;
  description @2 :Text;
  entrypoint @3 :Text;
  code @4 :Text;
  tests @5 :Text;
  capabilities @6 :ToolchainCapabilitiesPayload;
  requirements @7 :List(Text);
}

struct ToolchainTestResultPayload {
  name @0 :Text;
  version @1 :Text;
  ok @2 :Bool;
  output @3 :Text;
}

