type HealthResponse = {
  status: string;
  service: string;
  version: string;
};

async function fetchHealth(): Promise<HealthResponse | null> {
  const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
  const healthUrl = apiBase.endsWith("/api") ? `${apiBase}/health` : `${apiBase}/health`;

  try {
    const response = await fetch(healthUrl, { cache: "no-store" });
    if (!response.ok) return null;
    return (await response.json()) as HealthResponse;
  } catch {
    return null;
  }
}

export default async function Home() {
  const health = await fetchHealth();

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_0%_0%,#dceaf8_0%,#f2f4ef_45%),linear-gradient(120deg,#f2f4ef_0%,#e8ede5_100%)] px-6 py-12 text-[var(--foreground)] md:px-12">
      <main className="mx-auto max-w-6xl">
        <section className="rounded-3xl border border-[var(--line)] bg-[var(--card)]/85 p-8 shadow-[0_25px_80px_-45px_rgba(16,22,31,0.35)] backdrop-blur md:p-12">
          <p className="font-mono text-sm uppercase tracking-[0.25em] text-[var(--muted)]">
            No Code ML Platform
          </p>
          <h1 className="mt-4 max-w-4xl text-4xl font-semibold leading-tight md:text-6xl">
            Cloud-native ML infrastructure for serious production velocity.
          </h1>
          <p className="mt-6 max-w-3xl text-lg leading-relaxed text-[var(--muted)]">
            End-to-end stack with AutoGluon Tabular backend, typed APIs, persistent model metadata,
            containerized runtime, and Kubernetes-ready deployment primitives.
          </p>

          <div className="mt-10 grid gap-4 md:grid-cols-3">
            <article className="rounded-2xl border border-[var(--line)] bg-white p-5">
              <p className="font-mono text-xs uppercase tracking-widest text-[var(--muted)]">Backend Health</p>
              <p className="mt-3 text-2xl font-semibold">{health?.status ?? "unreachable"}</p>
              <p className="mt-2 text-sm text-[var(--muted)]">{health?.service ?? "API not reachable"}</p>
            </article>
            <article className="rounded-2xl border border-[var(--line)] bg-white p-5">
              <p className="font-mono text-xs uppercase tracking-widest text-[var(--muted)]">API Version</p>
              <p className="mt-3 text-2xl font-semibold">{health?.version ?? "n/a"}</p>
              <p className="mt-2 text-sm text-[var(--muted)]">FastAPI service runtime</p>
            </article>
            <article className="rounded-2xl border border-[var(--line)] bg-white p-5">
              <p className="font-mono text-xs uppercase tracking-widest text-[var(--muted)]">Training Engine</p>
              <p className="mt-3 text-2xl font-semibold">AutoGluon Tabular</p>
              <p className="mt-2 text-sm text-[var(--muted)]">Model artifacts + DB registry</p>
            </article>
          </div>

          <div className="mt-10 flex flex-wrap gap-3">
            <a
              href="/"
              className="rounded-xl bg-[var(--accent)] px-5 py-3 text-sm font-semibold text-white transition hover:bg-[var(--accent-strong)]"
            >
              Platform Console
            </a>
            <a
              href="http://localhost:8000/docs"
              className="rounded-xl border border-[var(--line)] bg-white px-5 py-3 text-sm font-semibold text-[var(--foreground)] transition hover:bg-[#eef2ed]"
            >
              API Docs
            </a>
          </div>
        </section>
      </main>
    </div>
  );
}
