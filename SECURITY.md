# Security policy

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private vulnerability reporting
(Security tab > "Report a vulnerability") on this repository. Expect an acknowledgement within
7 days. This is a pre-1.0 project maintained by one person, so there is no bug bounty and no
guaranteed fix timeline; confirmed issues are fixed on `main` and noted in release notes.

## Scope

In scope: the API server (auth, queue, TLS), `forgectl` and its node agent (TLS/mTLS, job
execution), the worker contract, the Dockerfile, Helm chart, and Terraform.

Known limits: the Docker image has not been rebuilt since later edits; the Terraform has only
been validated, never applied; Linux code paths are exercised in CI only. See `docs/networking.md`.
