# The instance pulls images from ECR and trained models from S3
resource "aws_iam_role" "app" {
  name = "${var.project_name}-ec2"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecr_read" {
  role       = aws_iam_role.app.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy" "models_read" {
  name = "${var.project_name}-models-read"
  role = aws_iam_role.app.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Resource = [aws_s3_bucket.models.arn, "${aws_s3_bucket.models.arn}/*"]
    }]
  })
}

resource "aws_iam_instance_profile" "app" {
  name = "${var.project_name}-ec2"
  role = aws_iam_role.app.name
}
