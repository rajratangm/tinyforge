# Worker contract v1alpha1

The boundary between the Go control layer (`forgectl`, node agent, operator) and the Python worker (`tinyforge`).
Go never touches tensors; it launches the worker and reads its output. Anything not listed here is not part of the contract.

## Invocation

    tinyforge worker run --spec /job/spec.yaml --out /job/out

- `spec.yaml` is a `TrainingJob` (see `jobspec.v1alpha1.schema.json`). The worker re-validates it and rejects unknown fields.
- `--out` is the job scratch directory. The worker writes only inside it (plus the model cache).
- Status: **not implemented yet.** Today the equivalent is `tinyforge ft train --json`; `worker run` will wrap it
  and map `spec` fields to `FTConfig` (`maxSteps`->`max_steps`, `loraR`->`lora_r`, `method: qlora`->`quant: 4bit`, ...).

## Output: JSON lines on stdout

One JSON object per line, flushed. Every object has `event` (string) and `t` (unix seconds). Human text goes to stderr.
Events already emitted by the worker today:

| event | fields | meaning |
|---|---|---|
| `started` | params, total_params, ... | training began |
| `resumed` | step | restarted from a checkpoint |
| `step` | step, loss, lr, grad_norm, tok_per_s, peak_mem_gb | every 5 steps |
| `eval` | step, val_loss, train_loss, val_ppl | every `eval_interval` |
| `diagnostic` | code, level (info/warn/error), message, fix | stable codes (FT001..., TR001...) |
| `finished` | steps, best_val_loss, peak_mem_gb, ... | success summary |
| `failed` | reason | terminal failure |

Planned additions: `gate` (metric, op, value, observed, passed), `artifact` (kind, path, sha256), `heartbeat` (every 30 s).
Consumers must ignore unknown events and unknown fields.

## Exit codes

| code | meaning | agent action |
|---|---|---|
| 0 | succeeded, all gates passed | mark Succeeded |
| 2 | spec invalid | mark Failed, do not retry |
| 3 | gate failed | mark Failed (artifacts kept), do not retry |
| 4 | out of memory / plan does not fit | mark Failed; retry only on a larger node |
| 5 | training diverged | mark Failed, do not retry |
| 75 | preempted (SIGTERM handled, checkpoint written) | reschedule; worker resumes |
| other | crash | retry per policy |

Codes 2-5 and 75 are proposals; the current worker exits 0/1 only. Align the worker before the agent depends on them.

## Signals and files

- `SIGTERM`: finish the current step, write an atomic checkpoint, emit `failed{reason:"preempted"}`, exit 75.
- Checkpoints are written atomically (`.tmp` then rename) to `out/` and resume is automatic when `checkpoint.resume` is true.
- Artifacts are listed by `artifact` events with sha256 so the registry can verify them. Safetensors only; never
  `torch.load(weights_only=False)` on anything not produced by the same job (known gap, OBJECTIVES M).

## Compatibility

`apiVersion` bumps on breaking changes; the worker must accept the previous version for one release.
