@0x89808d3353fbc573;

using Go = import "/go.capnp";
$Go.package("schema");
$Go.import("github.com/theapemachine/caramba/pkg/network/schema");

# ── Heterogeneous Hardware Profiling ──

struct ComputeProfile {
  availableRunners @0 :List(Text);   # e.g., ["cpu", "metal", "cuda", "xla"]
  vramBytes        @1 :UInt64;       # Video RAM available
  ramBytes         @2 :UInt64;       # System RAM available
  flopPerSec       @3 :UInt64;       # Estimated FLOP/s for routing score
}

struct NodeInfo {
  id      @0 :Data;           # 160-bit Kademlia Node ID
  address @1 :Text;           # IP:Port
  profile @2 :ComputeProfile; # Hardware profile for capability-based routing
}

# ── Distributed Compute Execution ──

struct Tensor {
  id    @0 :Text;
  shape @1 :List(Int64);
  data  @2 :Data;       # Zero-copy payload mapping directly to host/device memory
  dtype @3 :Text;       # "float64", "float32", "bfloat16"
}

struct IRNode {
  id      @0 :Text;
  op      @1 :Text;
  inputs  @2 :List(Text); # IDs of input nodes
  config  @3 :Data;       # JSON or MessagePack config payload
}

# ComputeStream demonstrates object capabilities and streaming.
# The client writes nodes and tensors continuously, then calls execute().
interface ComputeStream {
  writeNode @0 (node :IRNode) -> stream;
  writeTensor @1 (tensor :Tensor) -> stream;
  execute @2 () -> (
    outputs :List(Tensor),
    error :Text,
    workerAddress :Text,
    workerSignature :Data
  );
}

# The main RPC interface exposed by a Caramba network participant.
interface Peer {
  # ── Kademlia DHT ──
  ping @0 (sender :NodeInfo) -> (responder :NodeInfo);
  findNode @1 (sender :NodeInfo, targetId :Data) -> (responder :NodeInfo, nodes :List(NodeInfo));
  
  # ── Compute ──
  # startCompute returns a ComputeStream capability allowing the client
  # to stream data for a specific computation graph.
  startCompute @2 (graphId :Text) -> (stream :ComputeStream);
}
