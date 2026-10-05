#!/bin/bash
set -euo pipefail
mkdir -p /opt/tinyforge/{runs,data}

# API bearer token: read from SSM Parameter Store at boot, written to a root-only file, never put in user_data
# or on the docker command line. Without a token the API refuses /api/* (secure by default).
%{ if api_token_ssm_parameter != "" ~}
umask 077
token="$(aws ssm get-parameter --region ${region} --name '${api_token_ssm_parameter}' --with-decryption \
  --query Parameter.Value --output text)"
printf 'TINYFORGE_API_TOKEN=%s\n' "$token" >/opt/tinyforge/api.env
unset token
chmod 600 /opt/tinyforge/api.env
umask 022
%{ endif ~}

docker run -d --restart unless-stopped --gpus all --name tinyforge \
  -p 127.0.0.1:8000:8000 \
%{ if api_token_ssm_parameter != "" ~}
  --env-file /opt/tinyforge/api.env \
%{ endif ~}
  -v /opt/tinyforge/runs:/app/runs -v /opt/tinyforge/data:/app/data \
  ${image}

# Sync checkpoints/eval reports to S3 every 5 minutes so spot interruptions lose little work.
cat >/etc/cron.d/tinyforge-sync <<'EOF'
*/5 * * * * root aws s3 sync /opt/tinyforge/runs s3://${bucket}/runs --only-show-errors
EOF
