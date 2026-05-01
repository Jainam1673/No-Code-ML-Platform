# SRE Operations Runbook: No-Code ML Platform

This document defines the operational standards, reliability targets, and incident response procedures for the No-Code ML Platform.

---

## 📈 Service Level Objectives (SLOs)

We target **99.9% Availability** for the API Plane and **98% Success Rate** for the Training Plane.

| Service | Indicator (SLI) | Target | Window |
| :--- | :--- | :--- | :--- |
| **API Plane** | Availability (Successful requests / Total requests) | 99.9% | 28 Days |
| **API Plane** | Latency (P95 of `/v1/models/*/predict`) | < 500ms | 28 Days |
| **Training Plane** | Quality (Successful jobs / Total submitted) | 98.0% | 28 Days |
| **Training Plane** | Freshness (Max queue delay) | < 2 mins | Rolling 1h |

---

## 🔍 Monitoring & Observability

### Health Endpoints
*   `/livez`: Process liveness.
*   `/readyz`: Comprehensive dependency check (PostgreSQL + Redis).
*   `/metrics`: Prometheus format exports.

### Critical Prometheus Queries (PromQL)
*   **API Error Rate:** `rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])`
*   **P99 Latency:** `histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))`
*   **Queue Backlog:** `celery_queue_length{queue="training"}`

---

## 🛠️ Change Management & Deployment

### Deployment Strategy
1.  **Staging Validation:** All changes must pass CI and be validated in a pre-production environment.
2.  **Migration-First:** Database migrations (`make backend-db-upgrade`) must be successfully applied before rolling out new application pods.
3.  **Rolling Updates:** K8s Deployments use `RollingUpdate` with `maxSurge: 1` and `maxUnavailable: 0` to ensure zero-downtime.

### Rollback Trigger Criteria
*   API 5xx error rate > 1% for 3 consecutive minutes post-deploy.
*   P95 latency increases by > 50% from baseline.
*   `readyz` probes failing on > 20% of the fleet.

---

## 🚨 Incident Response (Tiered)

### Tier 1: Service Interruption (API Down)
1.  **Verify Probes:** Check if `readyz` is failing globally or on specific pods.
2.  **Upstream Check:** Validate PostgreSQL and Redis connectivity logs.
3.  **Immediate Action:** If a recent deploy is suspected, **Roll back immediately** (`kubectl rollout undo`).

### Tier 2: Degraded Performance (High Latency/Queue Lag)
1.  **Resource Saturation:** Check HPA status and pod CPU/Memory usage.
2.  **Worker Contention:** If queue length is rising, scale the worker pool:
    ```bash
    kubectl scale deployment worker --replicas=10
    ```
3.  **Database Locks:** Inspect PostgreSQL for long-running transactions or lock contention.

---

## 💾 Data Management & Disaster Recovery

### Backups
*   **PostgreSQL:** Automated daily snapshots with WAL archiving (PITR).
*   **Artifacts:** S3/Object Store versioning (or PVC snapshots) for model binaries.

### Recovery from DB Corruption
The platform maintains a **Shadow Registry**. If PostgreSQL is lost:
1.  Restore DB from most recent snapshot.
2.  Run the `sync-registry` utility (planned) to re-import model metadata from the `.json` files in the artifact store.

---

## ⚖️ Capacity Planning

*   **API Pods:** Baseline 2 replicas. Scale when average CPU > 60%.
*   **Worker Pods:** Baseline 2 replicas. Scale based on `celery_queue_length`.
*   **Storage:** Monitor PVC usage. Alert at 80% capacity. Auto-growth enabled on compatible CSI drivers.

---

## 🛡️ Security Operations

*   **Identity:** Access to production APIs requires valid JWT/Service tokens (see [ROLE_MATRIX.md](./ROLE_MATRIX.md)).
*   **Secrets:** Credentials must be rotated every 90 days.
*   **Audit:** All `POST/PUT/DELETE` operations are logged with `request_id` and `user_actor` context.
