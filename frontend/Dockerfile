FROM oven/bun:1.2.17 AS deps
WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

FROM oven/bun:1.2.17 AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN bun run build

FROM oven/bun:1.2.17 AS runner
WORKDIR /app
ENV NODE_ENV=production

RUN addgroup --system app && adduser --system --uid 10001 --ingroup app app

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/.next/standalone ./

USER app

EXPOSE 3000
CMD ["bun", "server.js"]
