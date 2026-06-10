# DocGraph Data Model

DocGraph separates evidence from curated knowledge:

```text
source -> episode -> chunk -> claim_evidence -> claim -> node/edge context
```

## Core objects

```text
source  = stable origin identity
episode = immutable snapshot/event from source
chunk   = searchable evidence piece
node    = canonical architecture/system thing
alias   = alternative name for a node
edge    = relationship between nodes
claim   = curated statement about node/edge
claim_evidence = link from claim to supporting/refuting chunk
proposal = requested graph mutation
commit   = accepted mutation history
```

## Shared-knowledge metadata

Nodes, edges, and claims can carry:

```text
visibility: local | shared | global | shared_candidate
finder_role: role that discovered it
audience_roles: roles that should see it
interface_tags: why it may cross role boundaries
```

Meaning:

```text
local            = internal detail for finder_role unless audience_roles says otherwise
shared           = verified cross-role/interface/config/flow knowledge
global           = broad system-level knowledge
shared_candidate = likely cross-role impact, but not fully proven yet
```

The graph is one shared system graph. Do not duplicate the same claim per role. Use `audience_roles` and evidence links.

## Cross-role trigger examples

Use shared/global/shared_candidate when a finding touches:

```text
configuration
registers/fields
status/interrupts
memory layout
public API/interface
input/output channels
data path position
timing/frame/vsync behavior
test behavior used by users
build/generated headers
logs/debug/runbooks
```

## Proof path

```text
claim -> claim_evidence -> chunk -> episode -> source
```

## Retrieval path

```text
query -> aliases/nodes/claims/chunks -> anchors -> role-aware context packet
```

Role-aware retrieval returns role-local + shared + global + relevant shared_candidate knowledge. It filters another role's local implementation details unless the requesting role is explicitly in `audience_roles`.

Each context request records compact operator telemetry in `retrieval_runs`, including final IDs and `trace_json` for anchor decisions, optional semantic promotion, bounded expansion, and timing. This trace is diagnostic metadata, not evidence and not durable system knowledge.
