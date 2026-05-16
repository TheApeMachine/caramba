# Voluntary Distributed Compute Requirements

This document defines the implementation contract for opt-in distributed compute in Caramba. The goal is a research grid where participants donate idle machine time, earn internal compute credits for verified useful work, and later spend those credits for priority when they need distributed capacity.

Credits are scheduling rights inside the Caramba network. They are not research results, backend correctness proof, or artifact validity. The Notary decides artifact validity from evidence. The credit system decides who receives scarce compute first.

## Completion Standard

Voluntary distributed compute is complete only when all of the following hold:

1. A node can explicitly opt in, declare resource limits, and revoke participation.
2. A node advertises verifiable compute capabilities, not just self-reported labels.
3. Researchers submit jobs as manifests with exact resource, backend, data, privacy, and verification requirements.
4. The scheduler ranks jobs by locked credits, earned contribution history, age, resource fit, and Notary policy.
5. Workers earn credits only for verified useful work.
6. Requesters spend credits through escrow before execution begins.
7. Settlement records worker identity, requester identity, manifest hash, graph hash, input hashes, output hashes, verification procedure, resource accounting, and payout.
8. Malicious, incorrect, unavailable, or low-quality workers lose reputation and locked stake.
9. Private data, model weights, and unreleased artifacts are never exposed to workers unless the manifest explicitly allows it.
10. The same distributed execution path respects backend legality, tensor residency, and provenance requirements.

## Current Anchors

The current useful pieces are:

- `pkg/network/dht`: peer discovery and hardware profile advertisement.
- `pkg/network/transport`: Cap'n Proto peer RPC and compute stream transport.
- `pkg/backend/compute/orchestrator/network_runner.go`: remote graph execution entry point.
- `pkg/notary`: identities, signatures, ledger balances, artifact records, manifest submission, settlement.
- `pkg/qpool`: local job scheduling, backpressure, load balancing, retries, worker accounting.
- `docs/architecture.md`: cluster, actor, orchestrator, scheduler, backend split.
- `docs/notary.md`: ledger evidence model and query-time truth.

Current distributed execution must grow from "stream graph to one peer" into a verified multi-node compute market with capability proofs, job escrow, credit pricing, execution receipts, replication, challenge checks, and priority scheduling.

## Core Actors

Required actors:

| Actor | Responsibility |
| --- | --- |
| Volunteer Node | Donates bounded idle resources and signs work receipts |
| Requester | Submits manifests and locks credits for execution |
| Scheduler | Matches jobs to capable workers and assigns priority |
| Notary | Verifies evidence, settles credits, and answers validity queries |
| Verifier | Recomputes, spot-checks, compares, or audits claimed outputs |
| Storage | Holds content-addressed manifests, inputs, outputs, checkpoints, logs |
| Capability Oracle | Maintains measured hardware and backend capability attestations |

Every actor emits evidence. Only the Notary records settlement and validity conclusions.

## Node Opt-In Contract

Create `pkg/cluster/node`.

An opted-in node must declare:

- identity public key,
- reachable transport address,
- supported runners,
- backend capability hashes,
- CPU architecture and ISA features,
- GPU type and memory,
- RAM limit,
- disk cache limit,
- network egress limit,
- allowed job classes,
- allowed artifact visibility classes,
- available time windows,
- thermal and battery policy,
- maximum concurrent jobs,
- stake amount,
- revocation policy.

Node participation must be explicit in configuration loaded through `pkg/config`. A node must never donate compute because a network listener exists.

## Capability Attestation

Create `pkg/cluster/capability`.

Node capability advertisement must be measured and signed. A valid capability attestation contains:

```text
node identity
backend location
operation coverage hash
backend version hash
device profile
memory limits
benchmark suite hash
benchmark results
parity suite hash
parity results
timestamp
expiration
signature
```

The scheduler can only place jobs on nodes whose attestations satisfy the job's backend, dtype, layout, operation, state, memory, and privacy requirements.

Hardware labels such as `cuda`, `metal`, and `xla` are routing hints. They do not prove legality. Legality comes from signed backend capability contracts and verified tests.

## Job Manifest Contract

Distributed jobs must be manifest-declared:

```yaml
distributed:
  mode: volunteer_grid
  resources:
    backend: metal
    min_vram_bytes: 17179869184
    max_wall_time: 30m
    replicas: 2
  privacy:
    data_visibility: public
    weights_visibility: public
    output_visibility: requester
  verification:
    method: replicated_hash
    spot_checks: 8
    tolerance: exact_or_declared_ulp
  credits:
    max_spend: 5000
    priority_bid: 300
```

Required fields:

- backend capability requirements,
- resource limits,
- expected graph hashes,
- input artifact hashes,
- output artifact policy,
- verification method,
- credit budget,
- priority bid,
- failure policy,
- cancellation policy,
- checkpoint policy,
- privacy class.

## Credit Model

Create `pkg/economy/credit`.

Credits are earned by verified useful work:

```text
earned = useful_work_units
       * backend_scarcity_multiplier
       * correctness_multiplier
       * availability_multiplier
       * latency_multiplier
       * reputation_multiplier
```

Useful work units must be derived from measured execution evidence:

- backend event timing,
- operation mix,
- tensor byte movement,
- memory pressure,
- graph segment complexity,
- successful checkpoint count,
- verification outcome.

Self-reported FLOP/s is never paid directly. It is a scheduling feature, not settlement evidence.

Credits are spent by escrow:

```text
requester balance -> job escrow -> worker payout
```

If the job is cancelled before assignment, escrow returns to the requester. If a worker accepts a lease and fails its obligations, the worker stake is penalized according to the manifest and Notary policy.

## Priority Ranking

Create `pkg/cluster/scheduler`.

The scheduler ranks queued jobs with a deterministic score:

```text
priority =
  locked_credit_bid
  + earned_contribution_rank
  + queue_age_bonus
  + resource_fit_score
  + verification_confidence_score
  - risk_score
```

Required ranking properties:

1. Credits improve priority only when locked in escrow.
2. Donated verified compute improves future priority even before credits are spent.
3. Long-waiting jobs age upward to avoid permanent starvation.
4. Scarce hardware queues are ranked separately from common hardware queues.
5. Jobs with private data require eligible workers and may rank lower if fewer workers satisfy policy.
6. The ranking inputs are ledger-visible and reproducible.

"Cash in" means locking credits to raise priority for a specific manifest run. It does not override backend legality, privacy constraints, resource limits, or verification requirements.

## Worker Lease Protocol

Create `pkg/cluster/lease`.

A worker lease must contain:

- lease ID,
- requester identity,
- worker identity,
- job manifest hash,
- graph hash,
- input artifact hashes,
- assigned segment,
- resource limits,
- deadline,
- payout schedule,
- worker stake,
- verification method,
- cancellation terms,
- signatures from requester, worker, and scheduler.

Leases are revocable only by explicit cancellation, timeout, worker violation, requester violation, or Notary policy.

## Execution Receipt

Create `pkg/cluster/receipt`.

Every completed work unit produces a signed receipt:

```text
lease ID
worker identity
backend capability attestation
backend event timings
input hashes
output hashes
checkpoint hashes
log hash
memory high-water mark
host transfer count
kernel trace hash
verification payload hash
worker signature
```

Receipts become ledger evidence. They are not validity conclusions.

## Verification Requirements

Create `pkg/cluster/verification`.

Required verification methods:

- replicated execution with hash agreement,
- deterministic spot checks,
- scalar reference checks for small slices,
- backend parity probes,
- challenge tensors embedded by the scheduler,
- Merkleized output chunks,
- checkpoint hash validation,
- runtime trace validation,
- declared ULP comparison for floating point outputs.

The verifier must support operation-specific comparison contracts. Exact hashes are required for integer, token, metadata, and deterministic byte outputs. Floating point outputs use the operation's declared ULP contract.

Workers are paid only after the verification method declared by the job manifest succeeds.

## Security And Privacy Requirements

Create `pkg/cluster/sandbox`.

Volunteer nodes execute untrusted jobs. Requesters send potentially valuable manifests, inputs, and weights. Both sides need protection.

Required worker protections:

- CPU, memory, GPU, disk, and network limits,
- wall-clock deadline,
- artifact cache quota,
- process isolation,
- backend resource cleanup,
- signed job manifests,
- explicit allowed output sinks.

Required requester protections:

- privacy class enforcement,
- encrypted artifacts where supported,
- public-data-only routing by default,
- worker eligibility policy,
- no undeclared network egress,
- no undeclared artifact reads,
- signed execution receipts,
- reproducible verification.

Private training data and private weights require an explicit privacy mechanism before placement on volunteer nodes. Acceptable mechanisms include trusted owned nodes, enclave-backed workers, encrypted partial evaluation, synthetic challenge-only jobs, or manifest-declared public release.

## Scheduler Requirements

The distributed scheduler must:

- maintain hardware-specific queues,
- index workers by capability attestation,
- reserve credits before assignment,
- create signed leases,
- split large jobs into graph or data segments,
- prefer data locality when artifacts are large,
- replicate work according to verification policy,
- track worker availability,
- track worker reliability,
- handle checkpoint resume,
- record every scheduling decision as ledger evidence.

Scheduling must be deterministic from ledger snapshot, queue state, worker attestations, and manifest policies.

## Transport Requirements

Extend `pkg/network/schema/caramba.capnp` and `pkg/network/transport`.

Required transport capabilities:

- announce capability attestation,
- submit job intent,
- accept lease,
- stream graph segment,
- stream artifact chunk,
- stream checkpoint,
- stream receipt,
- stream verifier challenge,
- cancel lease,
- report progress,
- report resource usage.

The transport must support large artifacts through chunked content-addressed transfer. Graph and tensor payloads must be bounded by manifest limits before allocation.

## Notary Ledger Requirements

Extend `pkg/notary`.

Required ledger entry types:

- `NodeJoined`,
- `CapabilityAttested`,
- `JobSubmitted`,
- `CreditsEscrowed`,
- `LeaseCreated`,
- `LeaseAccepted`,
- `CheckpointRecorded`,
- `ReceiptSubmitted`,
- `VerificationCompleted`,
- `CreditsSettled`,
- `StakePenalized`,
- `JobCancelled`,
- `NodeRevoked`.

The Notary must answer:

- current credit balance,
- available escrow,
- earned contribution rank,
- worker reliability,
- job validity,
- receipt validity,
- artifact validity,
- queue priority proof.

## Abuse Resistance Requirements

The system must handle:

- Sybil identities,
- fake hardware claims,
- incorrect outputs,
- partial outputs,
- worker disappearance,
- requester disappearance,
- replayed receipts,
- duplicated outputs,
- colluding workers,
- benchmark gaming,
- data exfiltration attempts,
- denial-of-service jobs,
- priority spam.

Required controls:

- stake for worker leases,
- signed nonces on every lease and receipt,
- capability expiration,
- randomized challenges,
- replicated verification,
- per-identity queue limits,
- per-artifact transfer limits,
- reputation decay,
- Notary-visible scheduling evidence,
- quarantine for disputed workers and jobs.

## Acceptance Tests

Required test suites:

| Suite | Required proof |
| --- | --- |
| Opt-in | Nodes donate resources only after explicit configuration |
| Capability | Measured attestations gate scheduling |
| Credit escrow | Requester credits lock, settle, return, or penalize correctly |
| Priority ranking | Queue order is deterministic and ledger-explainable |
| Lease | Signed leases bind requester, worker, manifest, segment, and deadline |
| Receipt | Worker receipts include all hashes, timings, and signatures |
| Verification | Bad outputs fail and correct replicated outputs settle |
| Privacy | Private artifacts never route to ineligible workers |
| Transport | Large artifacts stream in bounded chunks |
| Scheduler | Jobs split, replicate, resume, and settle across multiple workers |
| Abuse | Replay, duplicate, fake capability, and disappearing worker cases are rejected |

Benchmarks must report scheduler latency, transport throughput, verification overhead, settlement overhead, worker utilization, queue wait time, and credit cost per verified work unit.

## Package Implementation Map

Required new packages:

- `pkg/cluster/node`,
- `pkg/cluster/capability`,
- `pkg/cluster/scheduler`,
- `pkg/cluster/lease`,
- `pkg/cluster/receipt`,
- `pkg/cluster/verification`,
- `pkg/cluster/sandbox`,
- `pkg/economy/credit`,
- `pkg/economy/priority`.

Required package rewrites:

- `pkg/network/dht`: advertise signed capability attestations, not plain runner strings.
- `pkg/network/transport`: carry leases, artifact chunks, challenges, receipts, progress, and cancellation.
- `pkg/backend/compute/orchestrator/network_runner.go`: assign graph segments through scheduler leases and verification policy.
- `pkg/notary`: record credit escrow, settlement, worker reputation, capability attestations, leases, receipts, and priority proofs.
- `pkg/qpool`: expose local worker capacity and resource usage to the cluster scheduler.
- `pkg/config`: declare opt-in policy, resource limits, privacy policy, and stake policy.

## Definition Of Done

The voluntary grid is usable when a researcher can:

1. Opt a node into donation with explicit resource and privacy limits.
2. See signed capability attestations for that node.
3. Submit a distributed manifest with credit budget and verification policy.
4. Watch the scheduler explain priority from ledger-visible inputs.
5. Run work across multiple volunteer nodes.
6. Verify outputs before settlement.
7. Earn credits for correct useful work.
8. Spend earned credits for higher priority on a later job.
9. Inspect all leases, receipts, challenges, payouts, penalties, and artifacts.
10. Reproduce the Notary's conclusion from ledger evidence.

That is the voluntary distributed compute contract.
