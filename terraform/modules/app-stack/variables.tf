variable "project_name" {
  type = string
}

variable "ec2_key_pair_name" {
  type = string
}

variable "ec2_instance_type" {
  type = string
}

variable "allowed_ssh_cidr" {
  type = string
}

variable "ec2_root_volume_gb" {
  type    = number
  default = 20
}

variable "route53_zone_name" {
  type    = string
  default = ""
}

variable "route53_subdomain" {
  type    = string
  default = "minifigures"
}
