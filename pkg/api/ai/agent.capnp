@0xd4c9c9f76e89a0d3;  # Unique file ID

using Go = import "/go.capnp";
$Go.package("ai");
$Go.import("github.com/theapemachine/caramba/pkg/api/ai");

# Import common types
using Import = import "../provider/provider.capnp";

# Agent interface
interface Agent {
  # Core agent methods
  process @0 (params :Import.ProviderParams) -> Import.ProviderParams;
  getName @1 () -> (name :Text);
  
  # Context management
  getContext @2 () -> Import.ProviderParams;
  setContext @3 (params :Import.ProviderParams) -> ();
  
  # Tool management
  addTool @4 (tool :Import.Tool) -> ();
  listTools @5 () -> (tools :List(Import.Tool));
}

# Configuration options for creating a new agent
struct AgentOptions {
  provider @0 :Import.Provider;
  context @1 :Import.ProviderParams;  # Initial context
}
