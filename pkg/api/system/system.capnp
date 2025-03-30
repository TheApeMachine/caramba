@0xd4c9c9f76e89a0d4;  # Unique file ID

using Go = import "/go.capnp";
$Go.package("system");
$Go.import("github.com/theapemachine/caramba/pkg/api/system");

# Import common types
using Import = import "../provider/provider.capnp";

# System tool interface
interface SystemTool {
  # Core methods
  process @0 (command :Command) -> (result :Result);
  
  # Agent and topic operations
  listAgents @1 () -> (agents :List(Agent));
  listTopics @2 () -> (topics :List(Topic));
  sendSignal @3 (signal :Signal) -> (result :Result);
  breakLoop @4 (agentId :Text) -> (result :Result);
  
  # Tool management
  getSchema @5 () -> Import.Tool;
}

# Command represents a system operation
struct Command {
  type @0 :Type;
  payload @1 :AnyPointer;  # Type-specific payload

  enum Type {
    listAgents @0;    # List all agents
    listTopics @1;    # List all topics
    sendSignal @2;    # Send a signal to agent/topic
    breakLoop @3;     # Break an agent's loop
  }
}

# Result of a system operation
struct Result {
  success @0 :Bool;
  error @1 :Text;
  data @2 :AnyPointer;  # Operation-specific result data
}

# Agent information
struct Agent {
  id @0 :Text;
  name @1 :Text;
  status @2 :Status;
  topics @3 :List(Text);  # Subscribed topics

  enum Status {
    running @0;
    stopped @1;
    error @2;
  }
}

# Topic information
struct Topic {
  name @0 :Text;
  subscribers @1 :List(Text);  # Agent IDs subscribed to this topic
  messageCount @2 :UInt64;
}

# Signal to send to agents/topics
struct Signal {
  type @0 :Type;
  targetId @1 :Text;     # Agent ID or topic name
  payload @2 :Text;      # Signal-specific data
  metadata @3 :List(Metadata);

  enum Type {
    stop @0;           # Stop an agent
    pause @1;          # Pause an agent
    resume @2;         # Resume an agent
    message @3;        # Send a message to topic
    custom @4;         # Custom signal type
  }
}

# Metadata key-value pair
struct Metadata {
  key @0 :Text;
  value @1 :Text;
}

# Configuration options for creating a new system tool
struct SystemToolOptions {
  bufferSize @0 :UInt32 = 1024;  # Default buffer size
  registry @1 :Registry;
}

# Registry for storing system commands
struct Registry {
  entries @0 :List(RegistryEntry);
}

struct RegistryEntry {
  name @0 :Text;
  command @1 :Command;
  timestamp @2 :Int64;  # Unix timestamp
} 