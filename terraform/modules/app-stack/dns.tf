# Optional: point <subdomain>.<zone> at the instance
data "aws_route53_zone" "zone" {
  count = var.route53_zone_name == "" ? 0 : 1
  name  = var.route53_zone_name
}

resource "aws_route53_record" "app" {
  count   = var.route53_zone_name == "" ? 0 : 1
  zone_id = data.aws_route53_zone.zone[0].zone_id
  name    = var.route53_subdomain
  type    = "CNAME"
  ttl     = 300
  records = [aws_instance.app.public_dns]
}
