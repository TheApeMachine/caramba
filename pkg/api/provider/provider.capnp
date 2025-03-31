@0xd4c9c9f76e89a0d2;  # Unique file ID

using Go = import "/go.capnp";
$Go.package("provider");
$Go.import("github.com/theapemachine/caramba/pkg/api/provider");

# Import common types
using Import = import "../tool/tool.capnp";

struct Message {
  role @0 :Text;
  content @1 :Text;
  name @2 :Text;
  reference @3 :Text;
  toolCalls @4 :List(ToolCall);
}

struct ToolCall {
  id @0 :Text;
  type @1 :Text;
  function @2 :ToolCallFunction;
}

struct ToolCallFunction {
  name @0 :Text;
  arguments @1 :Text;
}

struct ResponseFormat {
  name @0 :Text;
  description @1 :Text;
  schema @2 :Text;  # JSON schema as string
  strict @3 :Bool;
}

struct ProviderParams {
  model @0 :Text;
  messages @1 :List(Message);
  tools @2 :List(Import.Tool);
  responseFormat @3 :ResponseFormat;
  temperature @4 :Float64;
  topP @5 :Float64;
  topK @6 :Float64;
  frequencyPenalty @7 :Float64;
  presencePenalty @8 :Float64;
  maxTokens @9 :UInt32;
  stream @10 :Bool;
}

interface Provider {
  complete @0 (params :ProviderParams) -> ProviderParams;
  stream @1 (params :ProviderParams) -> ProviderParams;
  embed @2 (text :Text) -> (embedding :List(Float32));
}
