output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "ec2_instance_id" {
  value = aws_instance.app.id
}

output "ec2_instance_public_ip" {
  value = aws_instance.app.public_ip
}

output "s3_bucket_name" {
  value = aws_s3_bucket.models.id
}

output "domain_name" {
  value = one(aws_route53_record.app[*].fqdn)
}
