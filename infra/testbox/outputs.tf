output "instance_id" {
  value = aws_instance.box.id
}

output "bucket" {
  value = aws_s3_bucket.xfer.bucket
}

output "ssm_session" {
  value = "aws ssm start-session --target ${aws_instance.box.id} --region ${var.region}"
}
