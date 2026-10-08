output "ecr_repository_url" {
  value = module.app_stack.ecr_repository_url
}

output "ec2_instance_id" {
  value = module.app_stack.ec2_instance_id
}

output "ec2_instance_public_ip" {
  value = module.app_stack.ec2_instance_public_ip
}

output "s3_bucket_name" {
  value = module.app_stack.s3_bucket_name
}

output "domain_name" {
  value = module.app_stack.domain_name
}
