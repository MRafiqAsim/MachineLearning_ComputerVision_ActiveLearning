# Latest Amazon Linux 2023 image
data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

resource "aws_security_group" "app" {
  name        = "${var.project_name}-sg"
  description = "Web (Streamlit on 80) and SSH access"

  ingress {
    description = "Streamlit app"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "SSH for deployments"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "app" {
  ami                    = data.aws_ssm_parameter.al2023.value
  instance_type          = var.ec2_instance_type
  key_name               = var.ec2_key_pair_name
  vpc_security_group_ids = [aws_security_group.app.id]
  iam_instance_profile   = aws_iam_instance_profile.app.name
  user_data              = file("${path.module}/user-data.sh")

  root_block_device {
    volume_size = var.ec2_root_volume_gb
  }

  tags = {
    Name    = var.project_name
    Project = var.project_name
  }
}
