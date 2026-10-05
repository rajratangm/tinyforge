output "artifacts_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "gpu_instance_id" {
  value = try(aws_instance.gpu[0].id, null)
}

output "connect" {
  description = "Reach the UI with no open ports, via SSM port forwarding."
  value = try(
    "aws ssm start-session --target ${aws_instance.gpu[0].id} --document-name AWS-StartPortForwardingSession --parameters portNumber=8000,localPortNumber=8000",
    null
  )
}

output "kms_key_arn" {
  description = "Customer-managed key ARN, or null when the AWS-managed keys are used."
  value       = local.cmk_arn
}

output "cloudtrail_bucket" {
  description = "CloudTrail log bucket, or null when enable_cloudtrail is false."
  value       = try(aws_s3_bucket.trail[0].bucket, null)
}
