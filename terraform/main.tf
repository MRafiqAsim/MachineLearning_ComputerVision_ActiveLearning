# Infrastructure for the API + Streamlit app: one EC2 host running Docker, an ECR
# repository for the image, an S3 bucket for trained models and an optional DNS record.
#
#   terraform init && terraform apply -var="ec2_key_pair_name=my-key"
#
# For a shared remote state, add an S3 backend, e.g.:
#   terraform { backend "s3" { bucket = "<state-bucket>" key = "minifigure-vision.tfstate" region = "<region>" } }

provider "aws" {
  region = var.aws_region
}

module "app_stack" {
  source = "./modules/app-stack"

  project_name      = var.project_name
  ec2_key_pair_name = var.ec2_key_pair_name
  ec2_instance_type = var.ec2_instance_type
  allowed_ssh_cidr  = var.allowed_ssh_cidr
  route53_zone_name = var.route53_zone_name
  route53_subdomain = var.route53_subdomain
}
