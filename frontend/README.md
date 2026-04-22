# Frontend

Next.js frontend for the No-Code ML Platform.

This UI is intentionally lightweight and operationally focused: it surfaces platform status and acts as an entry point to backend API workflows.

## Toolchain Policy

- Package manager: `bun`
- Local development: `bun run dev`
- Production build and runtime: `bun run build` and `bun run start`

Do not introduce npm-based dependency workflows.

## Local Development

```bash
bun install --frozen-lockfile
bun run dev
```

Default URL:

- http://localhost:3000

Custom backend URL:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000 bun run dev
```

## Environment Variables

- `NEXT_PUBLIC_API_URL`: Base URL for backend health and API access used by the UI.

In Compose, this is typically set to a gateway-routed value.

## Build and Start

```bash
bun run build
bun run start
```

## Quality

```bash
bun run lint
bun run build
```

## Architecture Notes

- App Router structure under `app/`
- Server-rendered homepage fetches backend health for quick status feedback
- Global design tokens and typography defined in `app/globals.css` and `app/layout.tsx`

## Container Strategy

Frontend image uses multi-stage Docker build:

1. Dependency install via Bun
2. Next.js production build
3. Runtime stage serving standalone output with Bun

This keeps image size and runtime complexity lower while maintaining Bun-only parity.

## Deployment

- Local stack integration via root `docker-compose.yml`
- Kubernetes manifests in `infra/k8s/base/frontend.yaml`

