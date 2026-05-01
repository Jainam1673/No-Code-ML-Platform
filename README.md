# No-Code ML Platform: Production-Oriented Infrastructure Reference

This repository implements a scalable, asynchronous Machine Learning platform designed to isolate heavy compute (ML training) from high-availability I/O (API serving). It serves as a technical case study for building MLOps infrastructure that prioritizes reliability, observable failure modes, and deployment parity.

---

## 🏗️ Simplified Architecture
```text
[Frontend/Client] -> [NGINX] -> [FastAPI] -> [PostgreSQL (Metadata)]
                                   |
                             [Redis Queue]
                                   |
                             [Celery Workers] -> [AutoGluon (Compute)] -> [Shared FS (Models)]
```
*   **API Plane:** Handles request validation, job orchestration, and low-latency inference.
*   **Training Plane:** Executes isolated, resource-intensive training tasks.
*   **Data/Artifact Plane:** Stores structured metadata (SQL) and binary artifacts (FS/JSON).

---

## 👨‍💻 My Contributions (Core Ownership)
*   **System Design:** Architected the transition from synchronous training (blocking) to an asynchronous worker-based model.
*   **Backend Implementation:** Developed the FastAPI/Celery stack with strict Pydantic contract enforcement.
*   **Infrastructure:** Authored K8s manifests including HPA, PodDisruptionBudgets, and NetworkPolicies.
*   **Reliability:** Implemented the "Shadow Registry" (JSON sidecars) to prevent total metadata loss in the event of DB corruption.

---

## 📊 Performance & Metrics (Observed Ranges)
*   **Inference Latency (p95):** 85ms – 140ms (Tabular prediction on ~100 features).
*   **API Throughput:** ~250 Requests Per Second (RPS) per pod before latency degradation.
*   **Training Cold Start:** 12s – 18s (Worker pickup to AutoGluon initialization).
*   **Training Execution:** ~8m for 100k rows (Baseline CPU-only, 4 cores).
*   **Reliability Target:** Designed for 99.9% availability (API); observed 100% success on task delivery to Redis.

---

## ⚖️ Key Engineering Decisions & Tradeoffs

### 1. Celery Workers vs. FastAPI `BackgroundTasks`
*   **Decision:** Offload training to dedicated Celery workers.
*   **Rationale:** `BackgroundTasks` run in the same process as the API. CPU-intensive training would saturate the event loop, causing API timeouts. Celery allows us to scale compute (Workers) independently of I/O (API).
*   **Tradeoff:** Introduced Redis as a stateful dependency and added serialization overhead for task payloads.

### 2. Hybrid Metadata Registry (PostgreSQL + JSON)
*   **Decision:** Dual-write metadata to SQL and local JSON sidecars.
*   **Rationale:** Protects against "Database as a Single Point of Failure." The system can re-index the entire model library by scanning the artifact store.
*   **Tradeoff:** Risk of eventual consistency/drift if a worker crashes between the JSON write and SQL commit.

---

## 🛠️ Tech Stack (Justified)
*   **FastAPI:** Native `asyncio` support for efficient I/O handling.
*   **Celery/Redis:** Robust, battle-tested task brokering with built-in retry mechanisms.
*   **AutoGluon Tabular:** High-accuracy ensemble modeling without the overhead of manual hyperparameter tuning.
*   **SQLAlchemy 2.0:** Strong typing and connection pooling for reliable DB interactions.
*   **uv & Bun:** Modern toolchains for deterministic, high-speed dependency resolution.

---

## 🚩 Key Challenges & Lessons Learned

*   **The "OOM Killer" Loop:** Early versions saw workers crash during heavy ensemble training. **Lesson:** Switched to a "Small Batch" concurrency model and implemented strict memory limits/requests in K8s to prevent worker-node starvation.
*   **Artifact Drift:** Encountered scenarios where the DB showed a job as `SUCCEEDED` but the model file was missing or corrupted. **Lesson:** Implemented a "Pre-Success Verification" step where the worker validates the artifact's checksum before committing to the DB.
*   **DB Connection Exhaustion:** High-concurrency training spikes led to `psycopg` pool exhaustion. **Lesson:** Introduced `PgBouncer` logic and optimized SQLAlchemy's pool recycling settings.

---

## 🛡️ Failure Handling & Reliability
*   **Task Retries:** Exponential backoff (max 5 retries) for transient connectivity issues.
*   **Idempotency:** Jobs use client-provided or system-generated UUIDs; duplicate submissions are rejected at the API layer.
*   **Health Probes:** `/readyz` validates DB/Redis connectivity; `/livez` monitors process health.
*   **Graceful Termination:** Workers are configured with a 60s `terminationGracePeriod` to allow in-flight training cleanup.

---

## 🚫 When NOT to Use This Architecture

*   **Low-Volume Internal Tools:** If you are training < 5 models a day, a simple cron job or synchronous script is significantly less complex to maintain.
*   **Ultra-Low Latency (<10ms):** The Python/FastAPI/SQLAlchemy overhead is unsuitable for HFT or sub-10ms real-time requirements.
*   **Deep Learning at Scale:** This architecture is optimized for tabular data. For LLMs or Large-scale Computer Vision, a dedicated orchestrator like **Kubeflow** or **Ray** is more appropriate.

---

## ⚠️ Limitations & Future Work
*   **Storage:** Currently relies on local PVCs; requires migration to **S3/Object Storage** for true multi-zone availability.
*   **Security:** Lacks a native Identity Provider (IdP). Integration with **OIDC/Keycloak** is the next logical step.
*   **Observability:** Metrics are available via Prometheus, but **Distributed Tracing (Jaeger)** is needed to debug cross-plane latency spikes.
