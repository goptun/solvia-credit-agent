# VPS Infrastructure Pre-Flight Assessment

Task group 1 of `openspec/changes/add-solvia-foundation-mvp`. All checks below were performed **read-only** over SSH: process/resource inspection and HTTP requests to the already-running gateway service. No package was installed, no service was started/stopped/restarted, and no file was created or modified on the VPS. IPs, hostnames, usernames, and the domain in front of the existing reverse proxy are redacted below and replaced with placeholders, per this project's public-repo secrets policy.

## Method

- SSH to `<vps-host>` as `<vps-user>`, using a locally-held private key (never read, printed, copied, or referenced by anything other than its local path).
- Commands run: `lsb_release`/`uname`, `nproc`/`lscpu`, `free -h`, `swapon --show`, `df -h`, `uptime`, `docker --version`/`docker compose version`, `docker ps`, `docker stats --no-stream`, `ss -tlnp`, `ufw status verbose`, `docker network ls`, and a read of `nginx` vhost `server_name`/`ssl_certificate` directives (values only, not full configs).
- HTTP checks against the already-running `9router` gateway: `GET /v1/models`, `POST /v1/chat/completions` (plain completion, native tool-calling, and `response_format: json_object`) for both `solvia-fast` and `solvia-smart`.
- A local SSH tunnel (`ssh -L <local-port>:<gateway-host>:<gateway-port> <vps-user>@<vps-host>`) was established and torn down to verify local-development reachability.

## Findings

### OS / kernel / hardware
- Ubuntu 24.04.4 LTS, kernel 6.17 (Oracle Cloud ARM instance, aarch64/Neoverse-N1).
- 2 vCPU.
- 11 GiB RAM total, ~9.0 GiB available, **0 B swap configured**.
- 193 GB disk, 169 GB free (13% used).
- Uptime 13 days; load average 0.43 / 0.32 / 0.30 on a 2-vCPU box — largely idle.

### Docker
- Docker 29.8.0, Docker Compose v5.5.1.
- 9 containers already running, belonging to unrelated pre-existing personal projects (a newsletter API, a RAG/Qdrant stack, a fraud-detection stack with Kafka + Postgres, and the `9router` LLM gateway itself). Combined memory usage per `docker stats --no-stream`: **~1.9 GiB** of the 11 GiB total. CPU usage across all of them is negligible at idle.
- Several project-scoped bridge networks already exist (one per `docker compose` project); a new `solvia` project will get its own, as usual.

### Ports and firewall
- `ufw` is active, default-deny inbound, with only `22/tcp` and `80,443/tcp` allowed in (IPv4 and IPv6).
- The `9router` gateway container is published only on a private mesh-network interface (a Tailscale address), **not** on `0.0.0.0` and **not** covered by any `ufw` allow rule — it is not reachable from the public internet today. This matches the "gateway must never be exposed publicly" requirement without any change needed.

### Existing reverse proxy / TLS
- `nginx` 1.24.0 is already running and fronting the other projects with per-subdomain `server_name` vhosts under the box owner's personal domain, each with a statically-provisioned TLS certificate (`ssl_certificate` pointing at a local `.pem`) — TLS is **not** certbot-automated (`certbot.timer` is inactive).
- Conclusion for `design.md`: the future public demo (`add-vps-deploy`) should add one more subdomain vhost to this existing `nginx`, not introduce Caddy or Traefik.

### `9router` gateway verification
- `GET /v1/models` → **HTTP 200**, and the model list includes both `solvia-fast` and `solvia-smart` alongside the gateway's other configured models.
- Minimal chat completion → **HTTP 200** on both `solvia-fast` (resolved to a Gemini Flash-Lite model) and `solvia-smart` (resolved to a Gemini Flash "thinking" model).
- Native tool-calling (`tools` + `tool_choice: required`) → **HTTP 200** with a correct `tool_calls` response on both aliases.
- `response_format: json_object` → **HTTP 200** on both aliases.
- Local SSH tunnel → `GET /v1/models` through `http://localhost:<local-port>/v1` → **HTTP 200**, confirming the local-development access pattern in `design.md` works end-to-end.

**Two behavioral findings for the implementation, not just this assessment:**
1. `solvia-smart` resolves to a reasoning model that spends `reasoning_tokens` out of the same `max_tokens` budget *before* emitting visible content. A low `max_tokens` (tested at 10) produced an **empty completion with `finish_reason: "length"`** — indistinguishable from a truncation bug unless anticipated. Any `smart`-tier call (slot extraction, `financial_analyst`, `responder`) needs a generous `max_tokens` (e.g., ≥256), and the app should treat `finish_reason == "length"` with empty content as retryable, not a hard failure.
2. Even with `response_format: json_object` requested, `solvia-smart`'s raw content came back wrapped in a ```` ```json ... ``` ```` Markdown fence rather than bare JSON. The JSON-mode parsing fallback described in `design.md` (native tool-calling first, JSON-mode-parse-and-validate-and-retry fallback) must strip Markdown code fences before `json.loads`.

### Resource footprint estimate vs. headroom

Planned MVP stack for this change (per `proposal.md`/`design.md`): an `api` container (FastAPI + LangGraph, no embeddings model loaded yet — `fastembed` is unused until the RAG change) and a `postgres`+pgvector container. LangFuse stays on Cloud (not self-hosted) by default; the console and `crm_mock` are scaffold-only in this change, with no running container.

| Component | Est. memory | Est. CPU |
|---|---|---|
| `api` | 200–400 MiB | Low — I/O-bound on LLM calls over the tunnel/mesh network |
| `postgres` + pgvector | 150–300 MiB at this scale | Low |
| **New total** | **~350–700 MiB** | Low |

Against ~9.0 GiB currently available (and ~1.9 GiB already used by the four unrelated pre-existing projects), the new stack's estimated footprint leaves several GiB of headroom — a comfortable safety margin even if the existing projects see a traffic spike at the same time. CPU headroom is similarly ample (load average 0.43 on 2 vCPU, and the new workload is I/O-bound, not compute-bound). Disk (169 GB free) is not a constraint.

## Go/No-Go verdict: **GO**

No blocking risks. The plan in `design.md` (reuse the existing `nginx` reverse proxy, keep LangFuse on Cloud, no self-hosted extras in this change) matches what was actually found on the box, so no design changes are required before proceeding.

## Risks

1. **No swap configured (0 B)** — under simultaneous memory pressure from all five projects, the kernel OOM killer could abruptly kill a container instead of the system degrading gracefully. Low likelihood given current headroom (~9 GiB free), but cheap to mitigate.
2. **`solvia-smart` reasoning-token overhead** (see findings above) — must be handled in the LLM factory / response-parsing code (task group 4/7), not just noted here.
3. **Manual (non-certbot) TLS on the existing `nginx`** — adding the future public-demo subdomain vhost in `add-vps-deploy` will be a manual certificate step, not automatic renewal-and-issue.
4. **Tailscale is the only network boundary protecting the gateway today** — sufficient for this change (nothing here changes that exposure), but worth re-verifying in `add-vps-deploy` if the gateway's network configuration changes.

## Recommendations

- Reuse the existing `nginx` instance for the future public demo vhost rather than introducing Caddy/Traefik — already assumed in `design.md`, now confirmed compatible.
- Keep LangFuse on Cloud for this MVP (already the `design.md` default) rather than self-hosting, to avoid adding another Postgres + service to an already multi-tenant box.
- Set a generous `max_tokens` (e.g., ≥256) on every `smart`-tier call, and treat an empty completion with `finish_reason: "length"` as retryable rather than a hard error (task group 4).
- Strip Markdown code fences from any JSON-mode response before parsing (task group 4/7).
- Add per-container memory limits in the new `docker-compose.yml`, as good practice on a shared box, even though headroom is currently ample (task group 10).
- Optionally add a small swap file (e.g., 2 GiB) as cheap insurance against memory spikes across the box; not blocking for this change.
