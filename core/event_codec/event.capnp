@0xa1b2c3d4e5f60001;

# EventEnvelope Cap'n Proto schema
# Replaces JSON serialization for CCP events with zero-copy access.

struct EventEnvelope {
  # Unique event identifier (UUID hex string)
  id @0 :Text;

  # Unix timestamp (seconds since epoch, float64 for sub-second precision)
  ts @1 :Float64;

  # Event type identifier (e.g. "Message", "ToolResult", "Wake", "Idle")
  type @2 :Text;

  # Stable sender identity (agent/persona/user id)
  sender @3 :Text;

  # Priority: higher is more urgent (default 0)
  priority @4 :Int32 = 0;

  # Optional compute/latency budget in milliseconds (-1 = not set)
  budgetMs @5 :Int32 = -1;

  # Commitment lifecycle signal: -1 close, 0 neutral, +1 open
  commitmentDelta @6 :Int8 = 0;

  # Optional commitment id linking open/close pairs
  commitmentId @7 :Text;

  # Payload as opaque bytes (can be JSON, msgpack, nested Cap'n Proto, etc.)
  payload @8 :Data;
}

# Batch of events for efficient bulk transfer
struct EventBatch {
  events @0 :List(EventEnvelope);
}
