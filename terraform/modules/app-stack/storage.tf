data "aws_caller_identity" "current" {}

resource "aws_ecr_repository" "app" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the last 10 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10 }
      action       = { type = "expire" }
    }]
  })
}

# Trained models (data/models) are uploaded here and pulled by the instance on deploy
resource "aws_s3_bucket" "models" {
  bucket = "${var.project_name}-models-${data.aws_caller_identity.current.account_id}"
}
