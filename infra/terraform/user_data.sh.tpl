#!/bin/bash
set -euo pipefail
mkdir -p /opt/tinyforge/{runs,data}
docker run -d --restart unless-stopped --gpus all --name tinyforge \
  -p 127.0.0.1:8000:8000 \
  -v /opt/tinyforge/runs:/app/runs -v /opt/tinyforge/data:/app/data \
  ${image}

# Sync checkpoints/eval reports to S3 every 5 minutes so spot interruptions lose little work.
cat >/etc/cron.d/tinyforge-sync <<'EOF'
*/5 * * * * root aws s3 sync /opt/tinyforge/runs s3://${bucket}/runs --only-show-errors
EOF
