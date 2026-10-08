variable "aws_region" {
  type        = string
  description = "AWS region to deploy to."
  default     = "eu-west-3"
}

variable "project_name" {
  type        = string
  description = "Name prefix for all resources (ECR repository, S3 bucket, tags)."
  default     = "minifigure-vision"
}

variable "ec2_key_pair_name" {
  type        = string
  description = "Name of an existing EC2 key pair, used for SSH deployments."
}

variable "ec2_instance_type" {
  type        = string
  description = "EC2 instance type."
  default     = "t3.small"
}

variable "allowed_ssh_cidr" {
  type        = string
  description = "CIDR block allowed to SSH into the instance (restrict this to your IP)."
  default     = "0.0.0.0/0"
}

variable "route53_zone_name" {
  type        = string
  description = "Optional Route 53 hosted zone (e.g. example.com). Leave empty to skip DNS."
  default     = ""
}

variable "route53_subdomain" {
  type        = string
  description = "Subdomain to create in the hosted zone (e.g. minifigures)."
  default     = "minifigures"
}
