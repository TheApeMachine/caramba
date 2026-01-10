Complete MOSAIC Differentiable VM Architecture
This plan implements the five missing architectural elements discussed in the meeting notes to transform MOSAIC into a fully autonomous "Differentiable Virtual Machine."

Current State Analysis
What exists:

HomeostaticLoop, IntrinsicDrive, DriveBand in core/homeostasis.py - primitives only, not integrated
MosaicOpcode ISA in layer/mosaic/isa.py - defines 10 opcodes (NOP, READ_MEM, WRITE_MEM, CLEAR_MEM, IDLE, GATE_UP, GATE_DOWN, SCAN, COMMIT, RESPOND)
opcode_head in MemoryBlockLayer - emit-only, does not control behavior
sleep_replay_per_pair in datasets - training data exists, no runtime loop
EventResponder + ModelHandler in infer/event_runtime.py - reactive only (prompt-triggered)
mosaic_idle.yml preset - agent process exists but is a research loop, not dVM idle
What's missing:

Opcodes don't control registers/memory (just logged)
No impulse-driven awakening (only prompt-triggered)
No idle-time compute loop (model waits passively)
No medium/slow timescale learning (only fast writes)
No tool creation mechanism (only tool usage)
---

Phase 1: Wire Opcodes to Control Behavior
Currently opcodes are emit-only. Wire them to actually gate subsystem operations.

Files to modify:

layer/mosaic/block.py - Add opcode dispatch logic
Implementation:

# In _process_token or forward loop:
if self.opcodes_enabled:
    op_logits = self.opcode_head(u_t)
    op_id = op_logits.argmax(dim=-1)  # Hard decode

    # Use STE for training gradients
    op_soft = F.softmax(op_logits, dim=-1)
    op_hard = F.one_hot(op_id, self.opcode_vocab)
    op_sel = (op_hard - op_soft).detach() + op_soft

    # Gate operations based on opcode
    do_read = op_sel[:, MosaicOpcode.READ_MEM]
    do_write = op_sel[:, MosaicOpcode.WRITE_MEM] * write_gate
    do_clear = op_sel[:, MosaicOpcode.CLEAR_MEM]
Behavior mapping:

| Opcode | Effect |

|--------|--------|

| READ_MEM | Enable memory read path |

| WRITE_MEM | Gate write operation (AND with existing write_gate) |

| CLEAR_MEM | Zero target register/memory slot |

| GATE_UP/DOWN | Boost/suppress fusion gates |

| IDLE | Suppress output (internal consolidation) |

| COMMIT | Trigger commitment delta |

---

Phase 2: Integrate Homeostatic Impulses into Runtime
Connect HomeostaticLoop to the inference runtime so the model can self-activate.

Files to modify:

infer/event_runtime.py - Add impulse handling
New file: infer/autonomous_runtime.py - Idle loop with homeostasis
Key metrics to expose:

Memory utilization (write rate, occupancy)
Routing entropy
Commitment balance (open vs closed)
Output confidence (logit entropy)
Implementation skeleton:

@dataclass
class AutonomousRuntime:
    """Event runtime with self-activation via homeostatic impulses."""

    responder: EventResponder
    bus: EventBus
    homeostasis: HomeostaticLoop
    poll_interval_ms: int = 100

    async def run_loop(self):
        while True:
            # Check for external events first
            if self.bus.has_pending():
                event = self.bus.pop()
                self._handle_external(event)
            else:
                # Collect internal metrics
                metrics = self._collect_model_metrics()
                impulse = self.homeostasis.impulse(metrics)

                if impulse is not None:
                    # Self-activate: model generates based on impulse
                    self._handle_impulse(impulse)
                else:
                    # True idle: consolidation
                    await self._consolidate()

            await asyncio.sleep(self.poll_interval_ms / 1000)
---

Phase 3: Idle-Time Compute (Sleep/Consolidation)
Implement the "Wake-Sleep" cycle where idle time is used for memory consolidation.

Files to modify:

infer/autonomous_runtime.py (new) - Consolidation logic
data/mosaic_synth.py - Already has sleep_replay_per_pair
Consolidation operations:

Memory replay: Sample from mem_k/mem_v, predict associated values
State cleanup: Prune low-utility entries (based on mem_utility_head)
Gradient-free rehearsal: Run forward passes on cached patterns
async def _consolidate(self):
    """Run one consolidation step during idle time."""
    # Sample from memory
    replay_batch = self._sample_memory_replay()

    if replay_batch is not None:
        # Forward pass with IDLE opcode supervision
        with torch.no_grad():
            logits, aux = self.runner.forward_chunk(replay_batch)

        # Optional: gradient step if continuous learning enabled
        if self.online_optimizer is not None:
            self._online_step(logits, replay_batch)
---

Phase 4: Continuous Learning (Three Timescales)
Implement the three-timescale learning architecture discussed in meetings.

Timescales:

| Scale | Mechanism | When | What updates |

|-------|-----------|------|--------------|

| Fast | Memory writes | Every token | mem_k, mem_v, registers |

| Medium | Adapter consolidation | Every N tokens | LoRA/adapter weights |

| Slow | Full training | Offline | All parameters |

Files to create/modify:

New file: trainer/online.py - Medium-timescale online learning
infer/autonomous_runtime.py - Hook medium timescale
Implementation:

@dataclass
class OnlineLearner:
    """Medium-timescale adapter consolidation."""

    model: nn.Module
    adapter_params: list[nn.Parameter]  # Only update these
    optimizer: torch.optim.Optimizer
    update_interval: int = 1000  # tokens between updates
    replay_buffer_size: int = 10000

    def maybe_step(self, loss: Tensor, token_count: int):
        self.buffer.append((loss.detach(), token_count))

        if token_count % self.update_interval == 0:
            # Compute gradient on adapter params only
            for p in self.model.parameters():
                p.requires_grad = p in self.adapter_params

            # Aggregate recent losses
            recent_loss = torch.stack([l for l, _ in self.buffer[-100:]])
            recent_loss.mean().backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
---

Phase 5: Native Tool Building
Implement the mechanism for the model to create new tools, not just use predefined ones.

Approach: The model emits a structured "tool definition" event that gets validated and registered.

Files to create/modify:

New file: ai/tools/builder.py - Tool builder/registry
core/event.py - Add ToolDefinition event type
infer/event_runtime.py - Handle tool creation events
Tool Definition Schema:

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    implementation: str  # Python code or MCP endpoint
    sandbox: bool = True  # Run in sandbox by default

    def validate(self) -> list[str]:
        """Return list of validation errors, empty if valid."""
        errors = []
        if not self.name.isidentifier():
            errors.append(f"Invalid tool name: {self.name}")
        # ... more validation
        return errors
Event flow:

ToolRegistry
ToolBuilder
EventBus
Model
ToolRegistry
ToolBuilder
EventBus
Model
alt
[Valid]
[Invalid]
Emit ToolDefinition event
Handle event
Validate definition
Register new tool
Emit ToolRegistered event
Emit ToolRejected event
Safety constraints:

Sandboxed execution (no filesystem/network by default)
Rate limiting on tool creation
Human approval for privileged tools
Automatic deprecation of unused tools
---

Architecture Diagram
Tool System
Continuous Learning
MOSAIC Model
Autonomous Runtime
ToolBuilder
ToolRegistry
Fast: Memory Writes
Medium: Adapter Updates
Slow: Full Training
OpcodeHead
MosaicBlock
Memory
Registers
EventBus
HomeostaticLoop
ImpulseRouter
---

Testing Strategy
Each phase includes verification tests:

Opcodes: Unit test that READ_MEM opcode gates memory read contribution
Homeostasis: Test that high-entropy state triggers impulse event
Idle compute: Test that consolidation runs during no-event periods
Continuous learning: Test adapter weights change during inference
Tool building: Test that valid ToolDefinition creates usable tool