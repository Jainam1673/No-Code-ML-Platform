# Frontend

Next.js frontend for the No Code ML Platform.

## Tooling

- Package manager: Bun
- Build: Bun running Next.js build
- Runtime in container: Bun executing Next standalone server

## Local Development

```bash
bun install --frozen-lockfile
bun run dev
```

Default URL: http://localhost:3000

Custom backend URL:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000 bun run dev
```

## Production Build

```bash
bun run build
bun run start
```

## Container Build Strategy

- Multi-stage Docker build
- Bun dependency stage
- Bun build stage
- Bun runtime stage with Next standalone output

This keeps runtime footprint smaller while preserving Bun-only workflow in Docker.

## Deployment

- Local compose entrypoint is defined in the root docker-compose.yml.
- Kubernetes manifests are under infra/k8s/base.

