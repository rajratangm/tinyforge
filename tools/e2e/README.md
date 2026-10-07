# End-to-end check (tabular -> train -> evaluate -> serve)

Reproduces the run recorded in `benchmarks/e2e-titanic-3b-soup.json`:

1. `curl -L .../seaborn-data/master/titanic.csv -o titanic.csv`
2. `tinyforge data pii-scan` and `tinyforge data tabular titanic.csv --out titanic_sql.jsonl --count 1500`
3. a job spec with `backend: soup` (see `spec/examples/soup-8b-streaming.yaml`) whose `data.source` is that JSONL,
   run with `tinyforge worker run --spec job.yaml --out runs/e2e_titanic`
4. `python tools/e2e/eval_exec.py runs/e2e_titanic titanic.csv 60 out.json` (execution accuracy on the real table;
   `VARIANTS=base_instructed` adds the fair baseline)
5. `python tools/e2e/serve_check.py` (copy `titanic.csv` next to it): serves the adapter through `/v1` with guardrails.

Caveat: validation questions share templates and values with training questions, so this measures learning the SQL
task on one schema, not transfer to new tables.
