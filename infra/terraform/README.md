# tinyforge AWS infrastructure (Terraform)

A small, **dev-grade** stack: one artifacts bucket, one optional spot GPU instance reached only through SSM Session
Manager, and a cost budget. Defaults create **no GPU instance and no new paid service**.

> Status: `terraform fmt`, `init -backend=false`, `validate` and `tflint` (core ruleset) pass. **This configuration has
> never been planned or applied against a real AWS account.** Treat every change below as reviewed-but-unverified until
> you have run a plan.

## What changed in the hardening pass

| Change | Where | Cost / behaviour impact |
|---|---|---|
| S3 bucket policy denies non-TLS and TLS < 1.2 requests | `main.tf` `artifacts_tls` | free |
| Bucket ownership controls = BucketOwnerEnforced (ACLs disabled) | `aws_s3_bucket_ownership_controls` | free |
| `bucket_key_enabled` on SSE-KMS; abort incomplete multipart uploads after 7 days | encryption + lifecycle | slightly cheaper |
| Optional customer-managed KMS key with rotation for bucket and root volume | `use_customer_managed_key` (default **false**) | about USD 1/month/key + requests when on |
| Root volume size is a variable; `kms_key_id`, `delete_on_termination` explicit | `root_volume_gb` | none at default (100 GB) |
| IMDSv2 hop limit 1 (containers cannot read the instance role credentials) | `metadata_options` | none; the tinyforge container needs no AWS credentials |
| API token read from SSM Parameter Store at boot into a root-only file; never in user_data or the docker command line | `api_token_ssm_parameter`, `user_data.sh.tpl` | none; IAM limited to that one parameter |
| Optional real VPC/subnet instead of the default VPC | `vpc_id`, `subnet_id` | none by default |
| Security group `create_before_destroy` | `aws_security_group.gpu` | none |
| Forecasted-spend budget alert (100%) in addition to the 80% actual alert | `aws_budgets_budget` | free |
| Input validation on region, env, instance type, volume, email, budget, image, parameter name | `variables.tf` | none |
| Optional CloudTrail (multi-region, log validation, 90-day log expiry) | `enable_cloudtrail` (default **false**) | first management-event trail is free; S3 storage for logs |
| Optional GuardDuty | `enable_guardduty` (default **false**) | usage-billed after a 30-day trial |
| New outputs: `kms_key_arn`, `cloudtrail_bucket` (no secrets are output) | `outputs.tf` | none |

Already good and kept: IMDSv2 required, no inbound security group rules, encrypted root volume, SSE-KMS bucket with
versioning and Block Public Access, 30-day noncurrent-version expiry, SSM instead of SSH, spot by default.

## Still dev-grade (not done)

- **Default VPC and public subnet** unless you set `vpc_id`/`subnet_id`. For production use a private subnet and either NAT
  or VPC endpoints (S3, SSM, SSMMessages, EC2Messages, ECR, STS, Logs). Endpoints are not created here.
- **No S3 Object Lock.** It can only be set when a bucket is created; add it to a new bucket if you need WORM artifacts.
- **No remote state is created here.** The backend is a partial `backend "s3" {}`; the state bucket must already exist
  with versioning, encryption and Block Public Access.
- **Instance role bucket access is bucket-wide** (`GetObject`/`PutObject` on every key). Narrow it to prefixes once the
  object layout is final.
- **Egress is 0.0.0.0/0 on 443** (needed for Docker/ECR/Hugging Face/SSM). Use an egress allowlist or proxy for air-gapped
  or regulated setups.
- **Image tag defaults to `tinyforge:latest`**; pin an immutable tag or digest outside dev.
- **No ECR repository, WAF, ALB/TLS, CloudWatch/AMP, AWS Config, Security Hub, Inspector, multi-account landing zone or DR.**
- **Changing `user_data` on an existing instance** triggers a stop/start update; the GPU worker is off by default.
- With `use_customer_managed_key = true`, the identity running Terraform also needs `kms:CreateGrant` for EBS to use the key.

## Safe workflow (run these yourself; nothing here has been run for you)

```powershell
cd infra/terraform
terraform fmt -check -recursive
terraform init -backend=false      # local validation only, no AWS calls
terraform validate
```

Real environment (needs AWS credentials and an existing state bucket):

```powershell
terraform init `
  -backend-config="bucket=<state-bucket>" -backend-config="key=tinyforge/dev.tfstate" `
  -backend-config="region=<region>" -backend-config="encrypt=true" -backend-config="use_lockfile=true"
terraform plan -out tfplan -var="env=dev" -var="alert_email=you@example.com"
# Read the whole plan. Expect: bucket, ownership controls, TLS policy, IAM role/policy, security group, budget. No instance.
terraform apply tfplan
```

Turning things on, one at a time, with a plan between each:

- Store the token first: `aws ssm put-parameter --name /tinyforge/dev/api-token --type SecureString --value "<long random>"`,
  then `-var="api_token_ssm_parameter=/tinyforge/dev/api-token"`.
- GPU worker (spends money): `-var="enable_gpu_worker=true"`; use `use_spot=false` only if you need stability.
- `-var="use_customer_managed_key=true"`, `-var="enable_cloudtrail=true"`, `-var="enable_guardduty=true"`.

Do not enable CMK on an existing bucket that already holds objects without planning the re-encryption: existing objects
keep their old key until rewritten.

## What was and was not verified

Verified: `fmt -check`, `init -backend=false`, `validate` (type-checks every conditional branch), `tflint` core ruleset, and
the `user_data` template renders to syntactically valid bash with and without a token parameter.

Not verified: any real `plan` or `apply`; the AWS provider's runtime rules (for example CloudTrail bucket-policy
acceptance, KMS grants, SSM decryption through `aws/ssm`); the tflint AWS ruleset and checkov were not run; no cost
measurement.
