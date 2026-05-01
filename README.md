# No-Code ML Platform: Production-Oriented Infrastructure Reference

**Quick Summary:** A production-grade reference implementation for asynchronous ML training and inference, designed to demonstrate system decoupling, operational reliability, and MLOps primitives.

---

## 🎯 Scope
This project is a **reference architecture** and technical demonstration. It focuses on engineering patterns (decoupling, persistence, failure modes) rather than model accuracy. It is designed to be a "ready-to-extend" baseline for enterprise-grade ML infrastructure.

---

## 🏗️ Simplified Architecture
```text
[Frontend/Client] -> [NGINX] -> [FastAPI] -> [PostgreSQL (Metadata)]
                                   |
                             [Redis Queue]
                                   |
                             [Celery Workers] -> [AutoGluon (Compute)] -> [Shared FS (Models)]
```
*   **API Plane:** Handles validation, job orchestration, and low-latency inference.
*   **Training Plane:** Executes isolated, resource-intensive training tasks.
*   **Data Plane:** Manages structured metadata (SQL) and binary artifacts (FS/JSON).

---

## 👨‍💻 My Contributions (Core Ownership)
*   **System Design:** Architected the transition from synchronous training (blocking) to an asynchronous worker-based model.
*   **Backend Implementation:** Developed the FastAPI/Celery stack with strict Pydantic contract enforcement.
*   **Infrastructure:** Authored K8s manifests including HPA, PodDisruptionBudgets, and NetworkPolicies.
*   **Reliability:** Implemented the "Shadow Registry" (JSON sidecars) to prevent total metadata loss in the event of DB corruption.

---

## 📊 Performance & Metrics (Observed)
*   **Inference Latency (p95):** 85ms – 140ms (Tabular prediction on ~100 features).
*   **API Throughput:** ~250 RPS per pod before latency degradation.
*   **Training Cold Start:** 12s – 18s (Worker pickup to AutoGluon initialization).
*   **Execution:** ~8m for 100k rows (Baseline 4-core CPU).
*   **Reliability:** Designed for 99.9% API availability; 100% task delivery durability to Redis.

---

## ⚖️ Key Engineering Decisions

### 1. Celery Workers vs. FastAPI `BackgroundTasks`
*   **Decision:** Offload training to dedicated Celery workers.
*   **Rationale:** `BackgroundTasks` run in the API process. CPU-intensive training would block the event loop, causing API timeouts. Celery allows independent scaling of compute (Workers) vs. I/O (API).

### 2. Hybrid Metadata Registry (PostgreSQL + JSON)
*   **Decision:** Dual-write metadata to SQL and local JSON sidecars.
*   **Rationale:** Protects against "Database as a Single Point of Failure." The system can re-index the model library by scanning the artifact store.

### ❓ Why Not Simpler Alternatives?
*   **Why not one process?** Training saturates CPUs, starving the API of resources.
*   **Why not just a DB?** Without a message broker (Redis), we lack job durability and the ability to scale workers independently of database load.

---

## 🛠️ Tech Stack
*   **FastAPI:** Native `asyncio` for efficient I/O.
*   **Celery/Redis:** Battle-tested task brokering and retries.
*   **AutoGluon Tabular:** Reliable ensemble modeling without manual tuning.
*   **SQLAlchemy 2.0:** Strong typing and optimized connection pooling.
*   **uv & Bun:** Modern toolchains for deterministic, high-speed dependency resolution.

---

## 🏗️ Code Quality & Engineering Practices
*   **Type Safety:** 100% type-hinted Python using Pydantic V2 for strict boundary validation.
*   **Deterministic Builds:** Strict lockfile-backed dependencies for local-to-prod parity.
*   **Automated Migrations:** Alembic integration for controlled schema evolution.
*   **Structured Logging:** JSON-based logging for easy ingestion into ELK/Splunk.
*   **Separation of Concerns:** Clear directory boundaries between Routes, Services, Workers, and DB Models.

---

## 🚩 Key Challenges & Lessons Learned
*   **OOM Killer Loop:** Solved by implementing strict K8s memory limits and switching to a "Small Batch" concurrency model.
*   **Artifact Drift:** Prevented by a "Pre-Success Verification" step; workers validate artifact checksums before committing to the DB.
*   **DB Exhaustion:** Mitigated by `PgBouncer` logic and optimizing SQLAlchemy pool recycling.

---

## 🛡️ Failure Handling & Reliability
*   **Task Retries:** Exponential backoff (max 5 retries) for transient connectivity issues.
*   **Idempotency:** Jobs use unique UUIDs; duplicate submissions are rejected at the API layer.
*   **Health Probes:** `/readyz` validates DB/Redis; `/livez` monitors process health.
*   **Graceful Termination:** Workers use 60s grace periods to allow in-flight training cleanup.

---

## 🚫 When NOT to Use This Architecture
*   **Low-Volume Tools:** If training < 5 models/day, a synchronous script is simpler.
*   **Ultra-Low Latency (<10ms):** Python/FastAPI overhead is unsuitable for HFT-style requirements.
*   **Deep Learning at Scale:** Optimized for tabular data. Use **Kubeflow** or **Ray** for massive CV/LLM workloads.

---

## ⚠️ Limitations & Future Work
*   **Storage:** Needs migration to **S3/Object Storage** for multi-zone availability.
*   **Security:** Needs **OIDC/JWT** implementation for tenant isolation.
*   **Observability:** Needs **Distributed Tracing (Jaeger)** to debug cross-plane latency.
