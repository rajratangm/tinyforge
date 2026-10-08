# Ephemeral multi-GPU test box for tinyforge. Separate from ../terraform (the long-lived worker module):
# local state, SSM-only access (no inbound ports, no SSH keys), and a dead-man switch that powers the
# instance off (-> terminated) after max_hours so a forgotten box cannot drain credits.
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "tinyforge", Purpose = "testbox", ManagedBy = "terraform" }
  }
}

data "aws_caller_identity" "me" {}

locals {
  account_id = data.aws_caller_identity.me.account_id
  name       = "tinyforge-testbox"
}

data "aws_ssm_parameter" "dlami" {
  name = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
}

data "aws_vpc" "default" {
  default = true
}

# Transfer bucket: repo tarball in, results out. Private, encrypted, emptied on destroy.
resource "aws_s3_bucket" "xfer" {
  bucket        = "${local.name}-${local.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "xfer" {
  bucket                  = aws_s3_bucket.xfer.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "xfer" {
  bucket = aws_s3_bucket.xfer.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "box" {
  name_prefix        = "${local.name}-"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.box.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_iam_policy_document" "bucket" {
  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.xfer.arn]
  }
  statement {
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.xfer.arn}/*"]
  }
}

resource "aws_iam_role_policy" "bucket" {
  name   = "xfer-bucket"
  role   = aws_iam_role.box.id
  policy = data.aws_iam_policy_document.bucket.json
}

resource "aws_iam_instance_profile" "box" {
  name_prefix = "${local.name}-"
  role        = aws_iam_role.box.name
}

resource "aws_security_group" "box" {
  name_prefix = "${local.name}-"
  description = "tinyforge testbox: no inbound, HTTPS out"
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "HTTPS for packages, model downloads, SSM, S3"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    description = "HTTP for apt mirrors"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    description = "DNS"
    from_port   = 53
    to_port     = 53
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle { create_before_destroy = true }
}

resource "aws_instance" "box" {
  ami                                  = data.aws_ssm_parameter.dlami.value
  instance_type                        = var.instance_type
  iam_instance_profile                 = aws_iam_instance_profile.box.name
  vpc_security_group_ids               = [aws_security_group.box.id]
  instance_initiated_shutdown_behavior = "terminate"

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options { instance_interruption_behavior = "terminate" }
    }
  }

  metadata_options {
    http_tokens   = "required"
    http_endpoint = "enabled"
  }

  root_block_device {
    volume_size           = var.root_volume_gb
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  user_data = <<-EOT
    #!/bin/bash
    # Dead-man switch: power off (-> terminate) after max_hours no matter what.
    shutdown -h +${var.max_hours * 60}
  EOT

  tags = { Name = local.name }
}
