# System Architecture: No-Code ML Platform

## 🎯 Design Philosophy

The No-Code ML Platform is designed to solve the inherent conflict between **synchronous user interaction** (API requests) and **asynchronous, resource-intensive workloads** (ML training).

The architecture prioritizes:
*   **Decoupling:** Isolation of failures and resource contention between planes.
*   **Durability:** Multi-layered persistence of both metadata and model artifacts.
*   **Operational Visibility:** Native instrumentation for health, readiness, and performance metrics.
*   **Deterministic Infrastructure:** Reproducible build and deployment pipelines.

---

## 🏛️ Component Decomposition

### 1. API Plane (FastAPI)
*   **Role:** Acts as the entry point for all control operations (Job submission, Model metadata retrieval) and low-latency inference.
*   **Concurrency Model:** Asynchronous (ASGI) to handle high-volume I/O-bound requests while training is offloaded.
*   **Validation:** Uses Pydantic V2 for strict contract enforcement and automatic OpenAPI (Swagger) generation.

### 2. Training Plane (Celery + AutoGluon)
*   **Role:** Executes long-running, CPU/GPU-intensive training jobs.
*   **Distributed Task Queue:** Leverages Redis as a broker to manage task distribution across a pool of workers.
*   **Isolation:** Workers are stateless relative to the API; they communicate only through the Data Plane (PostgreSQL/Redis).

### 3. Data Plane (PostgreSQL + Redis)
*   **PostgreSQL:** The source of truth for all job states (`job_records`) and model provenance (`model_records`).
*   **Redis:** Serves as the transient messaging layer (Celery broker) and the result backend for task tracking.

### 4. Artifact Plane (Hybrid Registry)
*   **Filesystem Storage:** Persists binary model artifacts produced by AutoGluon.
*   **JSON Shadow Registry:** Each model generates a JSON metadata sidecar. This enables the system to rebuild the database state from the artifact store in disaster recovery scenarios.

---

## 🚀 Scalability Vectors

| Component | Scaling Strategy | Bottleneck |
| :--- | :--- | :--- |
| **API** | Horizontal (HPA on CPU/Request count) | DB Connection Pool |
| **Workers** | Horizontal (HPA on Queue Depth) | CPU/GPU availability |
| **PostgreSQL** | Vertical or Read-Replicas | Write throughput (IOPS) |
| **Redis** | Cluster/Sentinel | Memory (Task backlog size) |

---

## 🛡️ Reliability & Fault Tolerance

### Retry Policy & Idempotency
*   **Automatic Retries:** Celery tasks are configured with exponential backoff and jitter for transient failures (e.g., database connection blips).
*   **Job State Machine:** Jobs progress through `QUEUED -> RUNNING -> SUCCEEDED\|FAILED`. This state is persisted in PostgreSQL, allowing the API to resume status reporting even after a worker restart.

### Failure Mode Analysis (FMA)

| Failure | Impact | Mitigation |
| :--- | :--- | :--- |
| **Worker Crash** | Training job hangs/fails | Celery `acks_late` and visibility timeouts; Job timeout monitoring. |
| **PostgreSQL Outage** | Meta-data operations fail | API enters `unready` state (Readiness Probe fails); JSON registry preserves artifact metadata. |
| **Redis Outage** | Training queue stalls | API buffers requests if possible, or returns 503; Kubernetes restarts Redis. |
| **Artifact Store Full** | Training/Inference fails | HPA scaling limits; monitoring on volume usage; retention policies. |

---

## 🔒 Security Posture

*   **Runtime Isolation:** All containers run as non-privileged users.
*   **Network Segmentation:** Kubernetes `NetworkPolicies` ensure that only the API can communicate with the outside world, and only the Worker/API can talk to the database.
*   **Secret Management:** Externalized via Kubernetes Secrets; no credentials in the codebase or Docker images.

---

## 📈 Observability

*   **Health Probes:** `/livez` (liveness), `/readyz` (database/broker connectivity check), `/startupz`.
*   **Metrics:** Prometheus endpoint at `/metrics` tracking request latency, status codes, and job throughput.
*   **Logging:** Structured JSON logging for easy ingestion into ELK/Splunk stacks.
