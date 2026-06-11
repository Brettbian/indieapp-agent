# Enterprise Multi-Agent Orchestration & Compaction Engine

A production-grade, high-throughput backend architecture designed for stateful multi-agent coordination, intent-driven routing, and automated context compaction. 

This repository showcases the core AI-native orchestration engine extracted from a scaled multi-agent platform. It utilizes **LangGraph** for cyclic stateful flows, **gRPC** for low-latency streaming endpoints, and **Celery/Redis** for distributed asynchronous task execution across multiple LLM providers.

---

## 🏗️ System Architecture & Stateful Orchestration

The engine is built around a centralized State Graph that manages conversational state, evaluates intent at execution boundaries, and routes traffic dynamically to specialized worker agents.

```
                  ┌────────────────────────┐
                  │   gRPC Client Request  │
                  └───────────┬────────────┘
                              │ (Stream)
                              ▼
                  ┌────────────────────────┐
                  │    gRPC Server API     │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │      Intent Router     │
                  └───────────┬────────────┘
                              │
             ┌────────────────┼────────────────┐
             │ (Q&A / Chat)   │ (Creative / SE)│ (Iterative Edit)
             ▼                ▼                ▼
   ┌──────────────────┐┌──────────────┐┌──────────────┐
   │    Chat Agent    ││Generate Agent││  Edit Agent  │
   └─────────┬────────┘└──────┬───────┘└──────┬───────┘
             │                │               │
             └────────────────┼───────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │    Compaction Node     │◄─── (Token Threshold Gate)
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │    State Checkpoint    │───► [Postgres JSONB]
                  └────────────────────────┘
```

---

## 🛠️ Core Engineering & Innovations

### 1. Stateful Context Compaction Node (`app/services/compaction_node.py`)
To prevent context window token blowups and control inference costs, the engine introduces a custom state truncation gate inside the LangGraph cycle:
*   **Dynamic Token Budgeting:** Evaluates current chat state token volume using a cached thread-safe `tiktoken` encoder (`cl100k_base`).
*   **Sliding Window Extraction:** When state exceeds a pre-defined threshold (e.g., 50k tokens), the compaction node isolates older messages, compiles them into a semantic recursive summary, and prepends the summary while retaining the most recent messages.
*   **State Archives:** Summarized historical context is stored as immutable snapshots in PostgreSQL JSONB, ensuring zero loss of auditability.

### 2. Low-Latency gRPC Streaming Server (`app/api/grpc_service.py`)
Replaces standard HTTP endpoints with bidirectional gRPC streaming to achieve real-time response times for end-users:
*   Streamed state updates directly hook into LangGraph's node traversal.
*   Supports async generator loops to stream chunked LLM outputs immediately to client-side interfaces.

### 3. Asynchronous Task Processing (`app/core/celery_app.py`)
Heavy, latency-insensitive compute tasks (such as PowerPoint generation, multi-stage layout editing, or bulk S3 uploads) are offloaded from the main event loop to isolated Celery worker queues backed by Redis:
*   **Separation of Concerns:** `default`, `generate`, and `edit` queues operate on dedicated threads, preventing slow rendering tasks from blocking the low-latency chat loop.

---

## 📂 Repository Layout

```
├── app/
│   ├── api/
│   │   └── grpc_service.py         # gRPC endpoints & streaming state loop
│   ├── config/
│   │   └── llm_models.yaml         # Provider-agnostic model parameters & costs
│   ├── core/
│   │   ├── config.py               # Pydantic Settings & ENV validation
│   │   └── celery_app.py           # Distributed task queue definition
│   ├── proto/
│   │   └── ai_service.proto        # Client-facing gRPC contract definition
│   └── services/
│       ├── router_graph.py         # Stateful LangGraph cyclic workflow
│       ├── compaction_node.py      # Automated context compression & token gate
│       ├── chat_agent.py           # Specialized conversational agent
│       ├── generate_agent.py       # Layout generation coordinator
│       └── supervisor_agent.py     # Team-level state coordinator
├── pyproject.toml                  # Dependency declarations & package locks
├── Dockerfile                      # Production multi-stage deployment build
└── docker-compose.yml              # Local containerized infrastructure orchestrator
```

---

## 🔑 Technology Stack

*   **Orchestration:** LangGraph, LangChain, Custom State Machines
*   **Infrastructure:** gRPC (Protobuf), Celery, Redis, PostgreSQL (JSONB Checkpointer)
*   **APIs & Providers:** OpenAI GPT, Anthropic Claude, Google Gemini, AWS Bedrock
*   **Observability:** LangSmith, Structured Logging, custom Token Estimator

---

## 📜 MIT License
This project is licensed under the MIT License - open for technical portfolio review and academic demonstration.
