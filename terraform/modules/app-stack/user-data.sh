#!/bin/bash
# Install Docker on Amazon Linux 2023 and let ec2-user run it
set -ex
dnf update -y
dnf install -y docker
systemctl enable --now docker
usermod -aG docker ec2-user
