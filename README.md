# No-Code ML Platform: Production-Grade Infrastructure Reference

This repository implements a scalable, asynchronous Machine Learning platform designed to decouple resource-intensive training workloads from high-availability inference serving. It serves as a technical reference for building MLOps infrastructure that prioritizes reliability, observability, and deterministic deployment.

---

## 👨‍💻 My Contributions (Core Ownership)
As the sole architect and lead engineer, I owned the end-to-end lifecycle:
*   **System Design:** Architected the decoupled "Control Plane vs. Training Plane" model to prevent head-of-line blocking.
*   **Backend Implementation:** Developed the FastAPI/Celery/SQLAlchemy stack with a focus on type safety and async I/O.
*   **Infrastructure-as-Code:** Authored Kubernetes manifests including HPA, PodDisruptionBudgets, and NetworkPolicies for production-grade security and scaling.
*   **Reliability Engineering:** Designed the hybrid metadata registry and health-check strategy to ensure 99.9% API availability.

---

## 🏗️ Architecture Overview

The system is decomposed into four functional planes to isolate failure domains and resource contention:

1.  **API Plane (FastAPI):** High-concurrency control plane for job submission, metadata retrieval, and inference.
2.  **Training Plane (Celery + AutoGluon):** Distributed worker pool executing CPU/GPU-intensive ensemble learning.
3.  **Data Plane (PostgreSQL + Redis):** ACID-compliant metadata storage and low-latency task brokering.
4.  **Artifact Plane (Hybrid Registry):** Persistent storage for model binaries with redundant JSON metadata sidecars for disaster recovery.

---

## 🔄 End-to-End Request Flow
1.  **Submission:** Client POSTs training request to `/v1/models/train`.
2.  **Job Initialization:** API persists a `QUEUED` state in PostgreSQL and returns a unique `job_id`.
3.  **Broker Enqueue:** API dispatches a task to the Redis-backed Celery queue.
4.  **Asynchronous Training:** A worker pulls the task, marks it `RUNNING`, and executes AutoGluon Tabular training.
5.  **Artifact Persistence:** Worker writes the trained model and a JSON metadata snapshot to the shared volume.
6.  **Registry Update:** Worker marks the job `SUCCEEDED` in the DB and registers the new model.
7.  **Inference:** Client performs inference via `/v1/models/{model_id}/predict`, which loads the model into the API's local predictor cache.

---

## 🛠️ Tech Stack (Justified)
*   **FastAPI (Python 3.12):** Chosen for its native `asyncio` support and Pydantic-driven contract enforcement.
*   **Celery + Redis:** Industry standard for reliable, distributed task execution with robust retry logic.
*   **AutoGluon Tabular:** Provides state-of-the-art ensemble learning for tabular data with minimal configuration.
*   **SQLAlchemy 2.0:** Leverages modern Python typing for safe, performant ORM interactions.
*   **`uv` & `Bun`:** High-performance package managers to ensure bit-for-bit build reproducibility and fast CI cycles.
*   **Kubernetes (Kustomize):** Orchestrates scaling and provides the network/security abstractions required for enterprise deployments.

---

## ⚖️ Key Engineering Decisions & Tradeoffs

### 1. Celery Workers vs. FastAPI `BackgroundTasks`
*   **Decision:** Offload training to Celery workers.
*   **Rationale:** Training is CPU-bound. Using `BackgroundTasks` would block the API's event loop and lead to resource contention. Celery allows independent scaling of the worker pool based on queue depth.
*   **Tradeoff:** Increases architectural complexity and introduces Redis as a critical dependency.

### 2. Hybrid Metadata Registry (DB + JSON Sidecars)
*   **Decision:** Store model metadata in both PostgreSQL and local JSON files.
*   **Rationale:** Provides a "Shadow Registry." If the database is corrupted or lost, the system can rebuild its state by scanning the artifact store.
*   **Tradeoff:** Requires careful handling to prevent drift between the two stores (eventual consistency).

### 3. Migration-First Deployment Pattern
*   **Decision:** Enforce a K8s `Job` to run Alembic migrations before application pods rotate.
*   **Rationale:** Ensures the schema is always compatible with the incoming code version, reducing 5xx errors during rollouts.
*   **Tradeoff:** Slightly increases deployment duration.

---

## 🛡️ Failure Handling & Reliability
*   **Distributed Retries:** Celery tasks use exponential backoff with jitter to handle transient DB or I/O blips.
*   **Idempotency:** Training jobs are tied to UUIDs; workers check state before re-running to prevent duplicate training.
*   **Health Probes:** `/readyz` performs a "deep check" of PostgreSQL and Redis connectivity; `/livez` monitors process liveness.
*   **Graceful Shutdown:** Workers use SIGTERM handling to finish active training batches (up to 60s) before terminating.

---

## 🚀 Scalability & System Design Considerations
*   **Horizontal Scaling:** API pods scale on CPU/Request count via HPA; Workers scale based on the `celery_queue_length` metric.
*   **Anti-Affinity:** Kubernetes pod anti-affinity rules ensure that API and Worker pods are distributed across different nodes to prevent a single node failure from taking down the cluster.
*   **Network Isolation:** `NetworkPolicies` restrict traffic so the Frontend cannot communicate directly with the Database, enforcing strict layer separation.

---

## 💡 What This Project Demonstrates
*   **Senior Backend Engineering:** Mastery of async patterns, database concurrency, and distributed systems.
*   **Operational Maturity:** SRE-standard observability (Prometheus), runbooks, and disaster recovery planning.
*   **MLOps Proficiency:** Understanding the friction between data science workflows and production infrastructure.

---

## ⚠️ Limitations & Future Work
*   **Artifact Storage:** Current implementation uses local PVCs. Transitioning to **S3/GCS** is required for multi-region scalability.
*   **Security:** Uses basic environment-variable secrets. Needs integration with **HashiCorp Vault** or AWS/GCP Secret Managers.
*   **Authn/Authz:** Currently open API. Needs **OIDC/JWT** implementation for tenant isolation.
*   **Observability:** Prometheus is implemented; **OpenTelemetry** tracing is the next step for profiling training latency across distributed nodes.
