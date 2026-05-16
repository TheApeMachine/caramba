# Machine Learning Research Platform Requirements

This document defines the implementation contract for making Caramba a platform that a machine learning researcher can use as their primary architecture laboratory.

The target is exact: a researcher describes architecture, runtime behavior, state systems, training loops, evaluation, and instrumentation through manifests, then executes that system on every supported compute target with resident tensors, reproducible provenance, and measurable performance.

## Completion Standard

The platform is complete for research use only when all of the following hold:

1. Model topology, runtime loops, state objects, samplers, schedulers, training, evaluation, and instrumentation are manifest-declared.
2. Chat and diffusion are runtime programs over manifests, not hard-coded Go control loops.
3. KV cache, rotary position state, diffusion scheduler state, optimizer state, RNG state, and dataset cursor state are first-class runtime state objects.
4. Every state object is replaceable by manifest selection.
5. Graph execution keeps tensors resident on the selected backend until an explicit output boundary.
6. Every compute operation and optimizer has real implementations on Go scalar, AVX2, SSE2, NEON, Metal, CUDA, and XLA.
7. Backend legality is checked against exact operation, dtype, layout, state, and fusion capabilities.
8. Unsupported manifest behavior fails at compile time with the exact missing contract.
9. Research runs produce a signed provenance ledger containing manifests, source revisions, artifact hashes, backend capabilities, seeds, metrics, traces, checkpoints, and generated outputs.
10. Every claimed backend, runtime, and state feature has parity tests, integration tests, and benchmarks.

## Current Anchors

The current useful pieces are:

- `pkg/manifest`: YAML parsing, includes, variables, repeat expansion, static topology compilation.
- `pkg/chat/session.go`: terminal input loop.
- `pkg/chat/model.go`: model manifest compilation, token loop, KV and RoPE metadata binding, backend execution.
- `pkg/backend/compute/kv/cache.go`: current decoder KV store.
- `pkg/backend/compute/rotary/config.go`: current inverse-frequency schedules.
- `pkg/diffusion/pipeline.go`: prompt encode, scheduler loop, denoise, VAE decode, image write.

The missing piece is a typed runtime program layer. Static topology manifests describe tensor DAGs. They do not yet describe loops, state mutation, IO, sampling, scheduling, training, or inspection.

## Manifest System V2

The manifest compiler must compile two related contracts:

```text
system.topology -> compute graph
system.runtime  -> runtime program
```

`system.runtime` must declare assets, tokenizers, datasets, state objects, samplers, schedulers, runtime loops, graph calls, control predicates, IO streams, output artifacts, training steps, evaluation steps, telemetry taps, profiler scopes, checkpoint policy, and provenance policy.

The compiler must emit:

```text
manifest.Document
  GraphModules
  RuntimeProgram
  StateDeclarations
  AssetDeclarations
  CapabilityRequirements
  ProvenanceDeclarations
```

Validation is strict. A runtime step cannot reference an undeclared graph, value, state object, tokenizer, dataset, sampler, scheduler, output sink, backend capability, or artifact.

## Runtime Program IR

Create `pkg/runtime/program`.

Required types:

```go
type Program struct {
    Name   string
    State  []StateDeclaration
    Steps  []Step
    Graphs map[string]GraphModule
}

type Step struct {
    ID      string
    Op      OperationID
    Inputs  map[string]ValueRef
    Outputs map[string]ValueRef
    Config  map[string]any
    Body    []Step
}

type ValueRef struct {
    Namespace string
    Name      string
    Path      []string
}
```

Required runtime operation families:

| Family | Required operations |
| --- | --- |
| IO | `io.read_line`, `io.read_record`, `io.emit_text`, `io.emit_token`, `io.write_image`, `io.write_tensor`, `io.write_checkpoint` |
| Tokenizer | `tokenizer.encode`, `tokenizer.decode`, `tokenizer.stream_decode` |
| Control | `control.loop_count`, `control.loop_each`, `control.loop_until`, `control.break_if`, `control.continue_if` |
| Value | `value.assign`, `value.slice`, `value.append`, `value.clear` |
| Graph | `graph.call`, `graph.call_sequence`, `graph.bind_weights` |
| Sampler | `sampler.next_token`, `sampler.stop_matched`, `sampler.logprobs` |
| Scheduler | `scheduler.timesteps`, `scheduler.step`, `scheduler.scale_input` |
| State | `state.create`, `state.reset`, `state.append`, `state.update`, `state.branch`, `state.commit`, `state.inspect` |
| Training | `train.forward`, `train.loss`, `train.backward`, `train.optimizer_step`, `train.zero_grad`, `train.clip_grad` |
| Evaluation | `eval.metric`, `eval.compare`, `eval.record`, `eval.assert` |
| Telemetry | `telemetry.scope`, `telemetry.counter`, `telemetry.histogram`, `telemetry.trace_tensor` |

Create `pkg/runtime/executor`.

The executor must instantiate declared state, bind graph modules, bind assets, execute runtime steps, preserve backend tensor residency across graph calls, emit structured traces, and surface errors with program step ID, operation ID, manifest path, and referenced object name.

## Runtime Manifest Shape

A chat runtime must be expressible as YAML with these declared objects and step order:

```yaml
system:
  runtime:
    type: program
    entry: chat
    backend: metal
    assets:
      model: { source: openai-community/gpt2 }
      tokenizer: { source: openai-community/gpt2 }
    state:
      history: { type: token_buffer }
      position: { type: counter, initial: 0 }
      kv: { type: kv_cache, strategy: paged, layout: bhsd, residency: backend }
    samplers: { main: { type: categorical, temperature: 0.8, top_k: 50, top_p: 0.95 } }
    graphs: { forward: { topology: system.topology } }
    program:
      - { id: read_user, op: io.read_line, out: user_text }
      - { id: encode_user, op: tokenizer.encode, in: user_text, out: input_ids }
      - id: generate
        op: control.loop_count
        count: ${generation.max_new_tokens}
        body:
          - { id: forward, op: graph.call, graph: forward, inputs: { input_ids: input_ids, kv_cache: state.kv, position_start: state.position }, outputs: { logits: logits } }
          - { id: sample, op: sampler.next_token, sampler: main, logits: logits, history: state.history, out: next_token }
          - { id: append_history, op: state.append, target: state.history, value: next_token }
          - { id: emit, op: io.emit_token, tokenizer: tokenizer, token: next_token }
          - { id: stop, op: control.break_if, condition: sampler.main.stop_matched }
          - { id: carry_token, op: value.assign, target: input_ids, value: next_token }
          - { id: advance_position, op: state.update, target: state.position, update: increment }
```

A diffusion runtime must be expressible as YAML:

```yaml
system:
  runtime:
    type: program
    entry: image
    backend: metal
    state:
      latents: { type: tensor, init: gaussian, seed: ${generation.seed} }
      scheduler: { type: flow_match_euler, steps: ${generation.steps} }
    graphs:
      text_encoder: { manifest: model.diffusion.text_encoder }
      denoiser: { manifest: model.diffusion.denoiser }
      vae_decode: { manifest: model.diffusion.vae_decoder }
    program:
      - id: encode_prompt
        op: graph.call
        graph: text_encoder
        inputs: { input_ids: tokenizer.encode(prompt) }
        outputs: { prompt_embeds: prompt_embeds }
      - id: denoise_loop
        op: control.loop_each
        source: state.scheduler.timesteps
        as: timestep
        body:
          - { id: denoise, op: graph.call, graph: denoiser, inputs: { latents: state.latents, prompt_embeds: prompt_embeds, timestep: timestep }, outputs: { velocity: velocity } }
          - { id: scheduler_step, op: scheduler.step, scheduler: state.scheduler, latents: state.latents, velocity: velocity, timestep: timestep, out: state.latents }
      - { id: decode, op: graph.call, graph: vae_decode, inputs: { latents: state.latents }, outputs: { image: image } }
      - { id: write, op: io.write_image, input: image, path: ${generation.output} }
```

## State Object System

Create `pkg/runtime/state`.

Every state object must implement:

```go
type State interface {
    ID() string
    Type() string
    Reset(ctx context.Context) error
    Snapshot(ctx context.Context) (Snapshot, error)
    Restore(ctx context.Context, snapshot Snapshot) error
    Inspect(ctx context.Context) (Inspection, error)
}
```

Backend-bound state must implement:

```go
type BackendState interface {
    State
    Bind(ctx context.Context, backend BackendBinding) (BoundState, error)
}
```

Required state objects:

- `token_buffer`,
- `counter`,
- `kv_cache`,
- `tensor`,
- `rng`,
- `scheduler`,
- `optimizer`,
- `dataset_cursor`,
- `metric_accumulator`,
- `checkpoint_index`,
- `trace_buffer`.

State must be branchable so beam search, speculative decoding, tree search, classifier-free guidance, ablation forks, and recurrent experimental memories use the same runtime machinery.

## KV Cache Requirements

Rebuild `pkg/backend/compute/kv` around declared strategies.

Required interface:

```go
type Strategy interface {
    Name() string
    Plan(request PlanRequest) (Plan, error)
    Bind(ctx context.Context, backend BackendBinding, plan Plan) (Store, error)
}

type Store interface {
    Append(ctx context.Context, layer string, key Tensor, value Tensor) (View, error)
    View(ctx context.Context, layer string, span Span) (View, error)
    Branch(ctx context.Context) (Store, error)
    Commit(ctx context.Context) error
    Reset(ctx context.Context) error
}
```

Required strategies:

- `contiguous`: one persistent K and V allocation per attention layer.
- `paged`: fixed-size token pages with page table metadata.
- `sliding_window`: bounded window with resident overwrite policy.
- `sink_tokens`: persistent prefix tokens plus moving decode window.
- `quantized`: resident packed K and V with declared scale format.
- `compressed`: resident compression and decompression kernels.
- `branchable`: copy-on-write pages for beam and tree decoding.
- `shared`: named cache shared across graph modules.

Every strategy must declare dtype, layout, page or window size, maximum batch, maximum heads, maximum tokens, head dimension, residency, backend capability requirements, mutation semantics, and checkpoint encoding.

Attention operations must consume KV state through explicit graph inputs. Metadata injection in `pkg/chat/model.go` must be replaced by manifest-declared bindings.

## Rotary And Position Requirements

Rotary embeddings must be graph operations plus declared runtime position state.

Required implementations:

- default RoPE,
- Llama 3 RoPE scaling,
- linear scaling,
- dynamic NTK scaling,
- YaRN scaling,
- position interpolation,
- partial rotary dimensions,
- interleaved and half-split layouts,
- prefill position tensors,
- decode scalar position starts,
- resident sin/cos cache generation,
- resident RoPE application kernels for every backend.

`position_start` must be a runtime value. It must not be injected by scanning graph nodes in Go code.

## Chat Runtime Requirements

`pkg/chat` must become a command adapter over `pkg/runtime/executor`.

Required changes:

1. `Session.Run` only reads and writes terminal streams.
2. Generation behavior is loaded from a runtime manifest.
3. Prompt formatting, tokenization, sampling, stop policies, streaming, KV cache, RoPE position, and graph execution are runtime operations or state objects.
4. Token-by-token graph execution is a runtime loop.
5. The hard-coded loop in `ModelGenerator.Generate` is expressed by the runtime program compiler.

## Diffusion Runtime Requirements

`pkg/diffusion` must become a command adapter over `pkg/runtime/executor`.

Required runtime features:

- prompt encoding graph call,
- negative prompt graph call,
- classifier-free guidance,
- multi-encoder prompt conditioning,
- latent initialization,
- scheduler timestep iteration,
- scheduler input scaling,
- denoiser graph call,
- scheduler state update,
- VAE decode graph call,
- image write operation,
- intermediate image taps,
- latent checkpointing,
- seed-controlled reproducibility.

The denoising loop in `Pipeline.Generate` is expressed by the runtime program compiler.

## Training Requirements

Training must be manifest-driven through the same runtime program layer.

Required components:

- dataset declaration,
- dataloader state,
- tokenizer and image preprocessing programs,
- forward graph call,
- loss graph call,
- backward graph call,
- optimizer state,
- optimizer graph call,
- gradient accumulation,
- gradient clipping,
- mixed precision policy,
- checkpoint write and restore,
- validation loop,
- evaluation metrics,
- early termination predicates,
- telemetry and profiler scopes.

Optimizers must be compute operations with complete backend implementations. Optimizer state must be backend-resident and checkpointable.

## Backend Requirements

Every backend must expose a precise capability contract containing operation ID, dtype, layout, shape class, state object compatibility, mutation behavior, fusion forms, precision contract, benchmark identifier, and parity test identifier.

Graph lowering must validate against this contract. A graph is legal only when every operation, state binding, dtype, layout, and fusion has a real backend implementation.

Backend execution must provide resident parameter tensors, resident inputs, resident outputs, resident state, explicit upload boundaries, explicit readback boundaries, graph-level memory planning, graph-level command planning, kernel timing, allocation accounting, event traces, and deterministic error locations.

## Researcher Inspection Requirements

Researchers must be able to inspect and modify the system without writing Go code.

Required inspection features: graph visualization, runtime program visualization, tensor taps at any named value, activation statistics, KV cache inspection, scheduler trace inspection, optimizer state inspection, backend capability report, memory plan report, fusion report, operation timing report, manifest diff report, run comparison report, provenance ledger browser, and reproducibility report.

Required modification features: layer insertion, layer deletion, layer replacement, weight tying, LoRA insertion, adapter insertion, cache strategy replacement, scheduler replacement, sampler replacement, activation replacement, normalization replacement, branchable runtime state, and graph module reuse.

## Acceptance Tests

Required test suites:

| Suite | Required proof |
| --- | --- |
| Manifest runtime compiler | YAML program compiles into typed runtime IR with exact references |
| Runtime executor | Chat, diffusion, and training programs execute from runtime IR |
| KV strategies | Every strategy passes append, view, branch, commit, reset, checkpoint, and restore tests |
| RoPE | Every scaling variant matches scalar reference for prefill and decode |
| Chat | Terminal streaming runs from manifest program only |
| Diffusion | Scheduler loop runs from manifest program only |
| Training | A tiny model trains from manifest program only |
| Backend legality | Illegal backend contracts fail during lowering |
| Device residency | Backend integration tests detect host movement inside graph execution |
| Performance | Benchmarks exist for every operation, state strategy, graph call, and runtime loop |
| Provenance | Runs produce signed ledgers with all manifests and artifact hashes |

Backend kernel tests must run parity at `N = 1, 7, 64, 1024, 8192` plus operation-specific tensor shapes.

Benchmarks must include scalar reference, AVX2, SSE2, NEON, Metal, CUDA, XLA, memory allocation count, host transfer count, backend event timing, and graph-level wall time.

## Package Implementation Map

Required new packages: `pkg/runtime/program`, `pkg/runtime/compiler`, `pkg/runtime/executor`, `pkg/runtime/state`, `pkg/runtime/op`, `pkg/runtime/io`, `pkg/runtime/sampler`, `pkg/runtime/scheduler`, `pkg/runtime/training`, and `pkg/runtime/inspection`.

Required package rewrites: `pkg/chat` as a command and terminal adapter over runtime executor; `pkg/diffusion` as a command and image adapter over runtime executor; `pkg/backend/compute/kv` as a strategy-based backend-bound cache system; `pkg/backend/compute/rotary` as the full graph operation and backend kernel surface; `pkg/manifest` as graph plus runtime program compiler; `pkg/backend/compute/orchestrator` as legality checks over operation, state, dtype, layout, and fusion; `pkg/backend/compute/{cpu,metal,cuda,xla}` as complete operation and state contracts.

## Researcher Definition Of Done

A machine learning researcher can use the platform when they can:

1. Write a new transformer variant entirely as YAML.
2. Replace its KV cache strategy entirely as YAML.
3. Replace its RoPE scaling entirely as YAML.
4. Replace its sampler entirely as YAML.
5. Run it as chat without changing Go code.
6. Write a diffusion model runtime entirely as YAML.
7. Replace its scheduler entirely as YAML.
8. Train or fine-tune a model entirely as YAML.
9. Inspect graph, runtime, tensors, cache, scheduler, optimizer, timings, and provenance from the run artifact.
10. Move the same manifest across Go scalar, AVX2, SSE2, NEON, Metal, CUDA, and XLA with exact backend legality and proof output.

That is the platform contract.
