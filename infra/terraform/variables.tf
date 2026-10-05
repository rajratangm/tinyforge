variable "region" {
  type    = string
  default = "us-east-1"
}

variable "env" {
  type    = string
  default = "dev"
}

variable "enable_gpu_worker" {
  description = "Create the GPU instance. Leave false until you want to spend money."
  type        = bool
  default     = false
}

variable "instance_type" {
  type    = string
  default = "g4dn.xlarge" # 1x T4 16 GB
}

variable "use_spot" {
  type    = bool
  default = true
}

variable "container_image" {
  description = "Image the worker runs, e.g. <acct>.dkr.ecr.<region>.amazonaws.com/tinyforge:latest"
  type        = string
  default     = "tinyforge:latest"
}

variable "monthly_budget_usd" {
  type    = number
  default = 50
}

variable "alert_email" {
  type    = string
  default = ""
}
