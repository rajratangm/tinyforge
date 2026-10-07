# 2. Go for CLI/agent/operator, Python for the ML worker
Status: accepted (2026-10-04)

Context: the target is a Linux, Kubernetes-native platform from single GPU rigs to datacenters; ML code lives in Python.
Decision: `forgectl`, the node agent and the future operator are Go (static binaries, client-go ecosystem); training and
eval run as a Python worker behind a documented contract (`spec/`), driven by a versioned JobSpec.
Consequences: two toolchains and CI jobs; the JobSpec schema and worker contract are the stable seam and must be versioned.
