variable "region" {
  type    = string
  default = "us-east-1"

  validation {
    condition     = can(regex("^[a-z]{2}(-[a-z]+)+-[0-9]$", var.region))
    error_message = "region must look like us-east-1 or ap-south-1."
  }
}

variable "env" {
  description = "Environment name; part of the bucket and role names, so keep it short."
  type        = string
  default     = "dev"

  validation {
    condition     = can(regex("^[a-z0-9]([a-z0-9-]{0,18}[a-z0-9])?$", var.env))
    error_message = "env must be 1-20 chars: lowercase letters, digits and hyphens, not starting or ending with a hyphen."
  }
}

variable "enable_gpu_worker" {
  description = "Create the GPU instance. Leave false until you want to spend money."
  type        = bool
  default     = false
}

variable "instance_type" {
  type    = string
  default = "g4dn.xlarge" # 1x T4 16 GB

  validation {
    condition     = can(regex("^[a-z0-9]+\\.[a-z0-9]+$", var.instance_type))
    error_message = "instance_type must look like g4dn.xlarge."
  }
}

variable "use_spot" {
  type    = bool
  default = true
}

variable "root_volume_gb" {
  description = "Root EBS volume size (gp3, always encrypted)."
  type        = number
  default     = 100

  validation {
    condition     = var.root_volume_gb >= 50 && var.root_volume_gb <= 2000
    error_message = "root_volume_gb must be between 50 and 2000."
  }
}

variable "container_image" {
  description = "Image the worker runs, e.g. <acct>.dkr.ecr.<region>.amazonaws.com/tinyforge:<immutable-tag-or-digest>. Avoid :latest outside dev."
  type        = string
  default     = "tinyforge:latest"

  validation {
    condition     = length(trimspace(var.container_image)) > 0
    error_message = "container_image must not be empty."
  }
}

variable "api_token_ssm_parameter" {
  description = "Name of an existing SSM Parameter Store SecureString (encrypted with the AWS-managed aws/ssm key) holding the tinyforge API bearer token. The instance reads it at boot. Empty = no token is provided and the API refuses /api/* calls (secure by default)."
  type        = string
  default     = ""

  validation {
    condition     = var.api_token_ssm_parameter == "" || can(regex("^/?[A-Za-z0-9_.\\-/]+$", var.api_token_ssm_parameter))
    error_message = "api_token_ssm_parameter must be a valid SSM parameter name (letters, digits, . _ - /)."
  }
}

variable "monthly_budget_usd" {
  type    = number
  default = 50

  validation {
    condition     = var.monthly_budget_usd > 0
    error_message = "monthly_budget_usd must be greater than 0."
  }
}

variable "alert_email" {
  type    = string
  default = ""

  validation {
    condition     = var.alert_email == "" || can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", var.alert_email))
    error_message = "alert_email must be empty or a valid email address."
  }
}

# ---- Network placement (default: the account's default VPC, as before) --------------------------
variable "vpc_id" {
  description = "VPC for the security group. Empty = the default VPC (dev-grade). Set together with subnet_id for a real VPC."
  type        = string
  default     = ""
}

variable "subnet_id" {
  description = "Subnet for the GPU instance. Empty = AWS picks a default-VPC subnet. Use a PRIVATE subnet with NAT or VPC endpoints (S3, SSM, ECR, STS, Logs) for production."
  type        = string
  default     = ""
}

# ---- Optional hardening that costs money or changes key management: all OFF by default ----------
variable "use_customer_managed_key" {
  description = "Encrypt the artifacts bucket and root volume with a customer-managed KMS key (rotation on). Cost: about USD 1/month per key plus KMS requests. Default false = the AWS-managed aws/s3 and aws/ebs keys."
  type        = bool
  default     = false
}

variable "enable_cloudtrail" {
  description = "Create a CloudTrail trail (management events) and its log bucket. The first copy of management events per region is free; you pay S3 storage for the logs."
  type        = bool
  default     = false
}

variable "cloudtrail_retention_days" {
  description = "Days to keep CloudTrail logs in the log bucket."
  type        = number
  default     = 90

  validation {
    condition     = var.cloudtrail_retention_days >= 1
    error_message = "cloudtrail_retention_days must be at least 1."
  }
}

variable "enable_guardduty" {
  description = "Enable GuardDuty threat detection in this region. Billed by usage after the 30-day trial; see the AWS pricing page."
  type        = bool
  default     = false
}
