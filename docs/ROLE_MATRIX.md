# Domain Boundaries & Responsibility Matrix

This document defines the intersection of engineering domains within the No-Code ML Platform. It serves as a guide for understanding how different personas interact with the system and where their accountabilities lie.

---

## 🛠️ Software Engineering (SWE)
*Focus: API surface, business logic, and contract enforcement.*

*   **Accountabilities:**
    *   Implementing robust, typed request/response schemas (Pydantic).
    *   Maintaining clean service boundaries to prevent "leaky abstractions."
    *   Ensuring high unit test coverage and API contract stability.
*   **Key Files:** `backend/app/api/`, `backend/app/schemas/`, `backend/app/services/`.

---

## 🏗️ Platform & Infrastructure Engineering
*Focus: Container orchestration, CI/CD, and resource management.*

*   **Accountabilities:**
    *   Designing the Kubernetes baseline (HPA, PDB, NetworkPolicies).
    *   Managing the OCI build pipeline and artifact lifecycle.
    *   Implementing hardware-specific worker pools (e.g., GPU/TPU overlays).
*   **Key Files:** `infra/k8s/`, `Dockerfile`, `Makefile`, `docker-compose.yml`.

---

## 📈 Site Reliability Engineering (SRE)
*Focus: System availability, latency, and operational excellence.*

*   **Accountabilities:**
    *   Defining and monitoring SLOs/SLIs (Prometheus/Grafana).
    *   Developing runbooks for incident triage and disaster recovery.
    *   Optimizing resource utilization and deployment safety (probes, rollouts).
*   **Key Files:** `docs/OPERATIONS.md`, `backend/app/api/routes/health.py`.

---

## 🤖 Machine Learning Operations (MLOps)
*Focus: Training workflows, model provenance, and inference scaling.*

*   **Accountabilities:**
    *   Managing the asynchronous training lifecycle (Celery/Redis).
    *   Ensuring model artifact durability and metadata integrity.
    *   Implementing hardware-accelerated inference strategies.
*   **Key Files:** `backend/app/worker/`, `backend/app/services/model_service.py`.

---

## 📋 Responsibility Assignment (RACI)

| Component | SWE | SRE | Platform | MLOps |
| :--- | :---: | :---: | :---: | :---: |
| **API Logic** | **A**/R | I | C | C |
| **K8s Manifests** | C | **A** | R | I |
| **Training Engine** | C | I | C | **A**/R |
| **DB Migrations** | R | I | **A** | C |
| **Health Monitoring**| C | **A**/R | I | I |

*Legend: **A**=Accountable, **R**=Responsible, **C**=Consulted, **I**=Informed*

---

## 💡 Interview Positioning
This platform is designed to demonstrate **Cross-Functional Proficiency**. Candidates should use this repository to showcase their ability to "think across the stack"—from writing a type-safe FastAPI endpoint to diagnosing a Kubernetes OCI permission error or tuning a Celery task retry policy.
