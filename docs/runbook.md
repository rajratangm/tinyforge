# Runbook

One entry per failure mode. Format: symptom, check, fix. Only modes that exist in the code today are listed.

## API refuses to start / returns 503 "API token not configured"
Set `TINYFORGE_API_TOKEN` (production) or `TINYFORGE_AUTH=off` (local only). Never use `off` on a reachable network.

## API returns 429
Repeated wrong tokens from one client. Wait out the window or fix the client's token; a restart clears the counter.

## Job shows `interrupted` after a restart
By design: a job running at crash/restart is marked `interrupted` and not re-run. Resubmit it; training auto-resumes from the last atomic checkpoint.

## Worker exits with code 75
Retryable failure by the worker contract (`spec/`); the agent retries it. Other non-zero codes are final: read the job log.

## `forgectl doctor` / `plan` warns about VRAM
Follow the printed fix (smaller micro-batch, gradient checkpointing, 4-bit). Codes are listed in the README.

## `forgectl net check` fails
The failing check names the endpoint, protocol and error. Compare against allowed flows in `docs/networking.md` section 3.

## Known unhandled
Orphaned worker after hard-killing the agent (kill by PID manually); GPU Xid errors and node health are not monitored yet (Q3).
