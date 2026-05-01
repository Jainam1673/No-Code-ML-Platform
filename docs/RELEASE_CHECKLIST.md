# Release Governance & Quality Gates

This document outlines the mandatory procedures and quality gates for promoting code to production environments. Adherence to these gates ensures low-risk, predictable releases.

---

## 🚦 1. Quality Gates (Pre-Flight)

All changes must satisfy the following criteria before a Release Candidate (RC) is cut:

*   **Static Analysis:** `make backend-lint` and `make frontend-lint` must pass with zero warnings.
*   **Type Safety:** `mypy` (Backend) and `tsc` (Frontend) must confirm 100% type coverage.
*   **Unit & Integration Tests:** 100% pass rate on `pytest` suite.
*   **Security Scanning:** OCI images must be scanned for vulnerabilities (CVEs) with a "Clean" or "Accepted Risk" status.
*   **Documentation:** All API changes must be reflected in the OpenAPI spec and relevant `/docs` files.

---

## 🏗️ 2. Deployment Sequence

The deployment follows a **Migration-First, Immutable Artifact** pattern.

1.  **Artifact Promotion:** Publish immutable, tagged OCI images to the registry.
2.  **Infrastructure Config:** Apply any `ConfigMap` or `Secret` updates.
3.  **Schema Migration:** Execute the `backend-migrate` Job.
    *   *Exit Criteria:* Migration Job must reach `Succeeded` state before proceeding.
4.  **Service Rollout:** Update Deployments (`backend`, `worker`, `frontend`).
    *   *Strategy:* `RollingUpdate` with health-aware readiness probes.
5.  **Traffic Cutover:** Update Ingress/Gateway configuration if necessary.

---

## 🧪 3. Post-Deploy Validation (PDV)

A release is not considered "Complete" until the following smoke tests pass in production:

| Test Case | Procedure | Expected Result |
| :--- | :--- | :--- |
| **Liveness** | `curl -f /livez` | `200 OK` |
| **Readyz** | `curl -f /readyz` | `200 OK` (All dependencies up) |
| **Lifecycle** | Submit training job | Job reaches `SUCCEEDED` state |
| **Inference** | Call `/predict` on new model | Valid JSON response with predictions |
| **Observability** | Check `/metrics` | Prometheus scraper can reach endpoint |

---

## 🔄 4. Rollback & Emergency Procedures

### Automatic Rollback Triggers
*   Pod crash-looping for > 2 minutes.
*   P95 Latency > 2s for more than 5% of requests.
*   Database connection pool exhaustion.

### Manual Rollback Command
```bash
# Undo the most recent deployment
kubectl rollout undo deployment/backend -n nocodeml
kubectl rollout undo deployment/worker -n nocodeml
kubectl rollout undo deployment/frontend -n nocodeml
```

---

## 📝 5. Release Closeout

Upon successful validation:
1.  **Tag Release:** Create a Git tag (e.g., `v1.2.3`).
2.  **Publish Notes:** Summarize user-facing changes and any technical debt introduced.
3.  **Monitor:** High-fidelity monitoring for 60 minutes post-release.
