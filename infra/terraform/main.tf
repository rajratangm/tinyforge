terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.67"
    }
  }
  # Remote state is configured per environment at init time (partial backend config), e.g.:
  #   terraform init \
  #     -backend-config="bucket=<state-bucket>" \
  #     -backend-config="key=tinyforge/terraform.tfstate" \
  #     -backend-config="region=<region>" \
  #     -backend-config="encrypt=true" \
  #     -backend-config="use_lockfile=true"   # native S3 state locking, needs Terraform >= 1.10
  # The state bucket must have versioning + encryption + Block Public Access and is NOT created here
  # (it must exist before the first init). See README.md. CI uses `init -backend=false`.
  backend "s3" {}
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "tinyforge", ManagedBy = "terraform", Env = var.env }
  }
}

data "aws_caller_identity" "me" {}

locals {
  account_id      = data.aws_caller_identity.me.account_id
  use_cmk         = var.use_customer_managed_key
  cmk_arn         = one(aws_kms_key.artifacts[*].arn)
  vpc_id          = coalesce(var.vpc_id, one(data.aws_vpc.default[*].id))
  has_token_param = var.api_token_ssm_parameter != ""
  token_param_arn = "arn:aws:ssm:${var.region}:${local.account_id}:parameter/${trimprefix(var.api_token_ssm_parameter, "/")}"
  trail_name      = "tinyforge-${var.env}"
  trail_bucket    = "tinyforge-${var.env}-trail-${local.account_id}"
}

# Latest AWS Deep Learning Base GPU AMI (Ubuntu 22.04) with NVIDIA drivers + Docker preinstalled.
data "aws_ssm_parameter" "dlami" {
  name = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
}

# Dev-grade default: the account's default VPC. Set var.vpc_id + var.subnet_id to use a real (private) VPC.
data "aws_vpc" "default" {
  count   = var.vpc_id == "" ? 1 : 0
  default = true
}

# ---- Optional customer-managed KMS key (var.use_customer_managed_key) ---------------------------
resource "aws_kms_key" "artifacts" {
  count                   = local.use_cmk ? 1 : 0
  description             = "tinyforge ${var.env}: artifacts bucket and GPU root volume"
  enable_key_rotation     = true
  deletion_window_in_days = 30
}

resource "aws_kms_alias" "artifacts" {
  count         = local.use_cmk ? 1 : 0
  name          = "alias/tinyforge-${var.env}"
  target_key_id = aws_kms_key.artifacts[0].key_id
}

# ---- Artifacts bucket: checkpoints, datasets, eval reports -------------------------------------
resource "aws_s3_bucket" "artifacts" {
  bucket        = "tinyforge-${var.env}-${local.account_id}"
  force_destroy = false
}

# Disable ACLs: the bucket owner owns every object and access is governed by policies only.
resource "aws_s3_bucket_ownership_controls" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule { object_ownership = "BucketOwnerEnforced" }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    bucket_key_enabled = true # fewer KMS requests, lower cost
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = local.cmk_arn # null = AWS-managed aws/s3 key
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "expire-old-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration { noncurrent_days = 30 }
  }

  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload { days_after_initiation = 7 }
  }
}

# Deny any request that is not TLS 1.2+.
data "aws_iam_policy_document" "artifacts_tls" {
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
  statement {
    sid       = "DenyOldTLS"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "NumericLessThan"
      variable = "s3:TlsVersion"
      values   = ["1.2"]
    }
  }
}

resource "aws_s3_bucket_policy" "artifacts" {
  bucket     = aws_s3_bucket.artifacts.id
  policy     = data.aws_iam_policy_document.artifacts_tls.json
  depends_on = [aws_s3_bucket_public_access_block.artifacts]
}

# ---- Instance role: SSM (no SSH needed) + least-privilege bucket access ------------------------
data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "gpu" {
  name               = "tinyforge-${var.env}-gpu"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.gpu.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Key used by SSM Parameter Store SecureStrings (only looked up when a token parameter is configured).
data "aws_kms_alias" "ssm" {
  count = local.has_token_param ? 1 : 0
  name  = "alias/aws/ssm"
}

data "aws_iam_policy_document" "bucket" {
  statement {
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }
  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn]
  }

  # Only with a customer-managed key: use it for the bucket's SSE-KMS objects.
  dynamic "statement" {
    for_each = local.use_cmk ? [1] : []
    content {
      actions   = ["kms:Decrypt", "kms:GenerateDataKey", "kms:DescribeKey"]
      resources = [local.cmk_arn]
    }
  }

  # Only when a token parameter is configured: read exactly that one parameter and decrypt it via SSM.
  dynamic "statement" {
    for_each = local.has_token_param ? [1] : []
    content {
      actions   = ["ssm:GetParameter"]
      resources = [local.token_param_arn]
    }
  }
  dynamic "statement" {
    for_each = local.has_token_param ? [1] : []
    content {
      actions   = ["kms:Decrypt"]
      resources = [data.aws_kms_alias.ssm[0].target_key_arn]
      condition {
        test     = "StringEquals"
        variable = "kms:ViaService"
        values   = ["ssm.${var.region}.amazonaws.com"]
      }
    }
  }
}

resource "aws_iam_role_policy" "bucket" {
  role   = aws_iam_role.gpu.id
  policy = data.aws_iam_policy_document.bucket.json
}

resource "aws_iam_instance_profile" "gpu" {
  name = "tinyforge-${var.env}-gpu"
  role = aws_iam_role.gpu.name
}

# ---- Network: no inbound at all. Access via SSM Session Manager port-forwarding ----------------
resource "aws_security_group" "gpu" {
  name_prefix = "tinyforge-${var.env}-"
  description = "tinyforge GPU worker: no inbound"
  vpc_id      = local.vpc_id

  egress {
    description = "outbound for package/model downloads and SSM"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle { create_before_destroy = true }
}

# ---- GPU worker (spot by default) ---------------------------------------------------------------
resource "aws_instance" "gpu" {
  count                  = var.enable_gpu_worker ? 1 : 0
  ami                    = data.aws_ssm_parameter.dlami.value
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id != "" ? var.subnet_id : null
  iam_instance_profile   = aws_iam_instance_profile.gpu.name
  vpc_security_group_ids = [aws_security_group.gpu.id]

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options { instance_interruption_behavior = "terminate" }
    }
  }

  metadata_options {
    http_tokens                 = "required" # IMDSv2 only
    http_endpoint               = "enabled"
    http_put_response_hop_limit = 1 # containers cannot reach the instance role credentials
  }

  root_block_device {
    volume_size           = var.root_volume_gb
    volume_type           = "gp3"
    encrypted             = true
    kms_key_id            = local.cmk_arn # null = AWS-managed aws/ebs key
    delete_on_termination = true
  }

  user_data = templatefile("${path.module}/user_data.sh.tpl", {
    image                   = var.container_image
    bucket                  = aws_s3_bucket.artifacts.bucket
    region                  = var.region
    api_token_ssm_parameter = var.api_token_ssm_parameter
  })

  tags = { Name = "tinyforge-${var.env}-gpu" }
}

# ---- Cost guardrail ------------------------------------------------------------------------------
resource "aws_budgets_budget" "monthly" {
  name         = "tinyforge-${var.env}"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  dynamic "notification" {
    for_each = var.alert_email == "" ? [] : [1]
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = 80
      threshold_type             = "PERCENTAGE"
      notification_type          = "ACTUAL"
      subscriber_email_addresses = [var.alert_email]
    }
  }

  dynamic "notification" {
    for_each = var.alert_email == "" ? [] : [1]
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = 100
      threshold_type             = "PERCENTAGE"
      notification_type          = "FORECASTED"
      subscriber_email_addresses = [var.alert_email]
    }
  }
}

# ---- Optional: CloudTrail (var.enable_cloudtrail, default off) -----------------------------------
resource "aws_s3_bucket" "trail" {
  count         = var.enable_cloudtrail ? 1 : 0
  bucket        = local.trail_bucket
  force_destroy = false
}

resource "aws_s3_bucket_public_access_block" "trail" {
  count                   = var.enable_cloudtrail ? 1 : 0
  bucket                  = aws_s3_bucket.trail[0].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "trail" {
  count  = var.enable_cloudtrail ? 1 : 0
  bucket = aws_s3_bucket.trail[0].id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "trail" {
  count  = var.enable_cloudtrail ? 1 : 0
  bucket = aws_s3_bucket.trail[0].id
  rule {
    id     = "expire-logs"
    status = "Enabled"
    filter {}
    expiration { days = var.cloudtrail_retention_days }
  }
}

data "aws_iam_policy_document" "trail" {
  count = var.enable_cloudtrail ? 1 : 0

  statement {
    sid       = "AWSCloudTrailAclCheck"
    actions   = ["s3:GetBucketAcl"]
    resources = [aws_s3_bucket.trail[0].arn]
    principals {
      type        = "Service"
      identifiers = ["cloudtrail.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceArn"
      values   = ["arn:aws:cloudtrail:${var.region}:${local.account_id}:trail/${local.trail_name}"]
    }
  }
  statement {
    sid       = "AWSCloudTrailWrite"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.trail[0].arn}/AWSLogs/${local.account_id}/*"]
    principals {
      type        = "Service"
      identifiers = ["cloudtrail.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "s3:x-amz-acl"
      values   = ["bucket-owner-full-control"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceArn"
      values   = ["arn:aws:cloudtrail:${var.region}:${local.account_id}:trail/${local.trail_name}"]
    }
  }
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.trail[0].arn, "${aws_s3_bucket.trail[0].arn}/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "trail" {
  count      = var.enable_cloudtrail ? 1 : 0
  bucket     = aws_s3_bucket.trail[0].id
  policy     = data.aws_iam_policy_document.trail[0].json
  depends_on = [aws_s3_bucket_public_access_block.trail]
}

resource "aws_cloudtrail" "main" {
  count                         = var.enable_cloudtrail ? 1 : 0
  name                          = local.trail_name
  s3_bucket_name                = aws_s3_bucket.trail[0].id
  include_global_service_events = true
  is_multi_region_trail         = true
  enable_log_file_validation    = true
  depends_on                    = [aws_s3_bucket_policy.trail]
}

# ---- Optional: GuardDuty (var.enable_guardduty, default off) -------------------------------------
resource "aws_guardduty_detector" "main" {
  count  = var.enable_guardduty ? 1 : 0
  enable = true
}
