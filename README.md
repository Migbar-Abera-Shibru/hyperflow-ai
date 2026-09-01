# HyperFlow AI

<div align="center">

**Schema-Aware Agent Planning with Tool-Schema Hypergraphs**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

##  What is HyperFlow AI?

HyperFlow AI is a **production-grade implementation** of the HyperAgent research paper, transforming cutting-edge academic research into a scalable, enterprise-ready agent planning engine.

Instead of the traditional **sequential, trial-and-error approach** used by ReAct agents, HyperFlow AI:

- **Constructs schema-aware hypergraphs** that model fine-grained tool dependencies at the parameter level
- **Plans globally** by generating Task DAGs with explicit data dependencies
- **Executes dynamically** using Deficit-Oriented Expansion for state-conditioned tool composition
- **Enables parallel execution** by identifying independent subtasks

Think of it as **"LangGraph meets Kubernetes for AI Agents"** — a complete planning and execution framework that understands exactly which outputs feed into which inputs, reducing redundant API calls and token consumption by **up to 40%**.


---

## Why This Matters?

### The Problem with Current Agents

Current LLM agents (ReAct, etc.) are **myopic** — they select tools sequentially, without understanding what's needed to make those tools work:

```bash
Agent: "I'll call send_payment"
→ Error: "recipient_id required"
Agent: "Let me get the user_id first"
→ Calls get_user
→ Success! Now call send_payment
```

text

This leads to:
- ❌ **Wasted API calls** — exploring dead ends
- ❌ **High token consumption** — repeated reasoning loops
- ❌ **No parallel execution** — sequential everything
- ❌ **Brittle** — fails when tools change

### The HyperFlow AI Solution

We model **data flow**, not just tool names:

```mermaid
graph TD
    A["getUser(user_id: int)<br/>Outputs: { user_id, name, email }"]
    B["send_payment(recipient_id: int,<br/>amount: float,<br/>currency: str)<br/>Outputs: { transaction_id, status }"]
    
    A -->|"user_id"| B
```
Now the agent knows:

- getUser must be called before send_payment

- The user_id output feeds directly into recipient_id

- These can be parallelized with other independent tasks

---

##  Key Features

###  Hypergraph Construction
- **Schema-level dependencies** — model which outputs satisfy which inputs
- **OpenAPI parsing** — automatically build hypergraphs from API specs
- **Semantic inference** — use embeddings to discover hidden dependencies
- **Manual refinement** — expert annotations for domain-specific knowledge

###  Schema-Aware Planning
- **Task interpretation** — LLM understands the task at a schema level
- **Context extraction** — backward expansion to recover prerequisite tools
- **DAG generation** — create directed acyclic graphs of subtasks
- **Dependency validation** — ensure data flows are complete and correct

###  Deficit-Oriented Expansion
- **State-conditioned** — adapts to what's already available
- **Beam search** — explores multiple compositions simultaneously
- **Deficit resolution** — automatically finds producers for missing inputs
- **Complete graphs** — ensures every required input has a producer

###  Dynamic Execution
- **Topological execution** — runs subtasks in dependency order
- **Parallel subtasks** — executes independent subtasks concurrently
- **State management** — Redis-backed agent state with bindings
- **Retry logic** — exponential backoff with configurable retries
- **Replanning** — adapts when subtasks fail

###  Observability
- **OpenTelemetry** — distributed tracing
- **Prometheus** — metrics collection
- **Grafana** — real-time dashboards
- **Structured logging** — JSON logs for analysis
- **Execution trace** — complete audit trail

---

##  Architecture
![Architecture Diagram](./images/arch.png)


### Data Flow

```mermaid
sequenceDiagram
    participant User
    participant API
    participant ContextExtractor
    participant DAGPlanner
    participant DOE
    participant Executor
    participant Redis
    participant LLM
    
    User->>API: POST /plan {"task": "Send $50 to Alice"}
    API->>ContextExtractor: extract_context(task)
    ContextExtractor->>LLM: Interpret task
    LLM-->>ContextExtractor: Task interpretation
    ContextExtractor->>ContextExtractor: Backward expansion
    ContextExtractor-->>API: Context Graph
    
    API->>DAGPlanner: plan(task, context)
    DAGPlanner->>LLM: Generate DAG
    LLM-->>DAGPlanner: Task DAG
    DAGPlanner->>DAGPlanner: Validate DAG
    DAGPlanner-->>API: ExecutionPlan
    API-->>User: Plan ID + DAG
    
    User->>API: POST /execute {plan_id}
    API->>Executor: execute(plan)
    
    loop For each subtask
        Executor->>DOE: expand(tool, state)
        DOE->>DOE: Compute deficits
        DOE->>DOE: Beam search producers
        DOE-->>Executor: Support Graph
        
        Executor->>Executor: Execute tools in order
        Executor->>Redis: Update state
        Executor->>Executor: Verify completion
        
        alt Needs replanning
            Executor->>DAGPlanner: refine_dag(dag, failure)
            DAGPlanner-->>Executor: Updated DAG
        end
    end
    
    Executor-->>API: ExecutionResult
    API-->>User: Results + metrics



