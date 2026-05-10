# Cap 'n Proto in Caramba

Caramba’s long-term architecture is **“objects sending messages”**. Most objects are intended to be **Cap’n Proto messages** (data) and **Cap’n Proto capabilities** (RPC), so that what starts as in-process composition can later become a **distributed system** with minimal conceptual change.

This doc explains what the “standard Cap’n Proto implementation” looks like in **Python** (schemas + runtime + RPC) and how it differs from the “generate code” workflow you may know from Go.

## Key idea: Python typically does *not* codegen

In Go, `capnp compile -ogo …` generates Go types.

In Python, the common library is **pycapnp**. With pycapnp you usually:

- **Write schemas** (`*.capnp`) as usual.
- **Load schemas dynamically** at runtime; Python gets **runtime-generated types**, not committed generated source files.
- Optionally run `capnp compile` in CI as a **schema validation step**, but not to produce Python code.

So: **you still use the Cap’n Proto compiler**, but usually as part of loading/validation rather than “codegen into Python files”.

## The Python stack: `capnp` (pycapnp) + the `capnp` compiler

Caramba uses:

- **`capnp` CLI**: the Cap’n Proto compiler (`capnp compile`, `capnp --version`)
- **`pycapnp`** (`import capnp`): Python bindings that load schemas and provide:
  - Struct builders/readers
  - Serialization (`to_bytes`, `from_bytes`, packed variants)
  - RPC and capabilities (asyncio + KJ loop integration)

The compiler is needed because pycapnp must compile the schema to a representation it can load. That compilation can happen:

- **Implicitly** (via pycapnp’s import hook, `import foo_capnp`)
- **Explicitly** (via `capnp.load("foo.capnp")`)
- **As a build/CI check** (via `capnp compile …` to ensure schemas are valid)

## Schema loading patterns in Python

### Option A: Import-hook “module per schema”

If `foo.capnp` exists on your Python path, you can do:

```python
import capnp  # required once to enable import hook
import foo_capnp

msg = foo_capnp.MyStruct.new_message()
msg.someField = "hi"
```

This is the most Go-like ergonomically: `foo_capnp` behaves like a generated module, but it’s created dynamically.

### Option B: Explicit load (no import hook)

If you want to avoid import-hook magic:

```python
import capnp

capnp.remove_import_hook()
foo_capnp = capnp.load("path/to/foo.capnp")
```

This is useful in tools, build scripts, and scenarios where you want very explicit control of schema locations.

## Serialization: messages are cheap to ship

Cap’n Proto messages can be serialized without expensive parsing. In Python you’ll commonly use:

- `msg.to_bytes()` / `Schema.Struct.from_bytes(data)`
- `msg.to_bytes_packed()` / `from_bytes_packed(data)` (smaller on wire)
- `msg.to_segments()` / `Schema.Struct.from_segments(segments)` (best fit for ZeroMQ multipart)
- File IO via `.write()` / `.read()` (and packed variants)

This fits Caramba’s “message passing” approach: even when components live in the same process, we try to keep their interfaces “message-shaped” so they can later be moved across process/network boundaries.

## RPC/capabilities in Python (the distributed part)

Cap’n Proto RPC is capability-based: an **interface** is a capability, and calling a method is a remote message send.

In Python (pycapnp), Cap’n Proto RPC is built on:

- **asyncio** (for Python scheduling)
- **KJ event loop** (Cap’n Proto’s underlying event loop)

### The required KJ loop context

All RPC calls must run inside the KJ loop context manager:

```python
import asyncio
import capnp

async def main():
    async with capnp.kj_loop():
        # create server/client and make RPC calls here
        ...

asyncio.run(main())
```

pycapnp also provides a helper `capnp.run(coro)` that runs a coroutine within the required KJ loop context.

### Server shape (Two-Party RPC)

The simplest deployment model is “two-party” RPC over a TCP connection:

- Server listens on a socket.
- Each new connection gets a `TwoPartyServer` bootstrapped with an implementation of your interface.

### Client shape

The client:

- Opens a connection via `capnp.AsyncIoStream.create_connection(...)`
- Creates a `TwoPartyClient`
- Bootstraps and casts the remote capability to your interface type

From there, method calls return awaitable promises.

## How this maps to Caramba’s architecture

Caramba aims to make **local composition** and **distributed composition** feel the same:

- **Structs** model the data flowing between subsystems (manifests, tasks, plans, metrics, topology, etc.).
- **Interfaces** model the active components (scheduler, runner, backend executor, storage, telemetry, etc.).

When everything speaks in Cap’n Proto messages:

- Swapping an in-process implementation for a networked one becomes mostly an infrastructure change.
- “Message passing” naturally encourages single-responsibility components (one request → one response).
- You can persist, replay, and audit interactions by storing serialized messages.

## Recommended project conventions (Caramba)

- **One type / concern per `.capnp` file**: keep files small and composable.
- **Compose via imports**: `using Foo = import "./foo.capnp".Foo;`
- **No silent defaults** at higher layers: if a field is required, validate it explicitly in Python.
- **Prefer explicit unions** over “Any”: if a field can be multiple things, model that as a union.
  - If you truly need “arbitrary JSON” for an extension point, store it as `Data` containing marshaled JSON 
    and validate it at boundaries.

## Build/CI: schema validation

Even though Python doesn’t typically check in generated code, it’s still valuable to ensure schemas compile:

```bash
capnp compile -oc++ path/to/schema.capnp
```

This is a **syntax/consistency check** and catches issues early (bad imports, invalid field numbers, etc.).

## References

- A2A Python SDK `AgentCard` type (for Caramba’s A2A schema mirroring): `https://a2a-protocol.org/latest/sdk/python/api/a2a.html#a2a.types.AgentCard`
- Cap’n Proto RPC overview: `https://capnproto.org/rpc.html`
- Cap’n Proto segments over ZeroMQ multipart (Kenton Varda): `https://stackoverflow.com/questions/32041315/how-to-send-capn-proto-message-over-zmq`