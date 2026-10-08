variable "region" {
  type    = string
  default = "us-east-1"
}

variable "instance_type" {
  description = "g5.12xlarge = 4x A10G 24 GB (48 vCPU, about USD 5.67/h on demand)."
  type        = string
  default     = "g5.12xlarge"
}

variable "use_spot" {
  description = "Spot is cheaper but has its own quota (L-3819A6DF) and can be interrupted mid-test."
  type        = bool
  default     = false
}

variable "max_hours" {
  description = "Dead-man switch: the instance powers off and is terminated after this many hours."
  type        = number
  default     = 8

  validation {
    condition     = var.max_hours >= 1 && var.max_hours <= 24
    error_message = "max_hours must be between 1 and 24."
  }
}

variable "root_volume_gb" {
  type    = number
  default = 200
}
