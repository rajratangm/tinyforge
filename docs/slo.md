# Service level objectives (draft)

| SLI | Source | Objective (proposed) |
|---|---|---|
| Job success ratio (done / (done + failed), interrupted excluded) | `tinyforge:job_success_ratio:30d` in `infra/monitoring/rules.yml` | >= 95% over 30 days |
| Queue drain: `tinyforge_queue_depth` <= 10 | `TinyforgeQueueBacklog` alert | no breach longer than 30 min |
| API scrapable | `up{job="tinyforge"}` | 99% monthly |

These are proposals for a single-operator deployment; no production traffic exists yet to calibrate them. The recording
rule is computed over all history in the exporter's database, not a true 30-day window, until counters replace the
queue-derived histogram. Rules are unit-tested (`promtool test rules infra/monitoring/rules_test.yml`).
