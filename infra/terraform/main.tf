terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  # Configure remote state per environment, e.g.:
  # terraform init -backend-config="bucket=<state-bucket>" -backend-config="key=tinyforge/terraform.tfstate"
  backend "s3" {}
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "tinyforge", ManagedBy = "terraform", Env = var.env }
  }
}

data "aws_caller_identity" "me" {}

# Latest AWS Deep Learning Base GPU AMI (Ubuntu 22.04) with NVIDIA drivers + Docker preinstalled.
data "aws_ssm_parameter" "dlami" {
  name = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
}

data "aws_vpc" "default" {
  default = true
}

# ---- Artifacts bucket: checkpoints, datasets, eval reports -------------------------------------
resource "aws_s3_bucket" "artifacts" {
  bucket = "tinyforge-${var.env}-${data.aws_caller_identity.me.account_id}"
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" }
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

data "aws_iam_policy_document" "bucket" {
  statement {
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }
  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.artifacts.arn]
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
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "outbound for package/model downloads and SSM"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---- GPU worker (spot by default) ---------------------------------------------------------------
resource "aws_instance" "gpu" {
  count                  = var.enable_gpu_worker ? 1 : 0
  ami                    = data.aws_ssm_parameter.dlami.value
  instance_type          = var.instance_type
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
    http_tokens   = "required" # IMDSv2 only
    http_endpoint = "enabled"
  }

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
    encrypted   = true
  }

  user_data = templatefile("${path.module}/user_data.sh.tpl", {
    image  = var.container_image
    bucket = aws_s3_bucket.artifacts.bucket
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
}
