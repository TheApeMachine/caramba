@0xd4c9c9f76e88a0d0;

using Go = import "/go.capnp";
$Go.package("agent");
$Go.import("github.com/theapemachine/caramba/pkg/ai/agent");

using Params  = import "../params/params.capnp";
using Context = import "../context/context.capnp";

struct Agent {
    identity @0 :Identity;
    params   @1 :Params.Params;
    context  @2 :Context.Context;
    tools    @3 :List(Text);
}

struct Identity {
    identifier @0 :Text;
    name       @1 :Text;
    role       @2 :Text;
}

struct Config {
    capabilities @0 :List(Text);     # What this agent can do
    requirements @1 :List(Text);     # What this agent needs before working
    behavior     @2 :Text;           # How the agent should behave (e.g. "planner", "developer")
    priority     @3 :UInt32;         # Task priority level
}

struct Task {
    id           @0 :Text;
    description  @1 :Text;
    status       @2 :TaskStatus;
    dependencies @3 :List(Text);     # IDs of tasks that must complete first
    assignedTo   @4 :Text;           # Agent identifier
    createdBy    @5 :Text;           # Agent identifier that created this task
    result       @6 :Text;
}

enum TaskStatus {
    pending    @0;
    blocked    @1;
    running    @2;
    completed  @3;
    failed     @4;
}

interface State {
    get @0 () -> (params :Params.Params, context :Context.Context);
    set @1 (params :Params.Params, context :Context.Context);
}

interface Tools {
    add @0 (name :Text);
    use @1 (name :Text, arguments :Text) -> (result :Text);
}

interface AgentRPC {
    # Core agent operations
    run     @0 () -> stream;
    stop    @1 () -> ();
    
    # Communication between agents - using byte arrays for serialized datura.Artifact
    process @2 (message :Data) -> (response :Data);
    
    # State management
    getState @3 () -> (state :Agent);
    setState @4 (state :Agent) -> ();
    
    # Tool operations
    asTool   @5 (name :Text, args :Text) -> (result :Text);
}