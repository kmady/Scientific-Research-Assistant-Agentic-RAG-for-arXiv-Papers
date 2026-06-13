output "alb_dns_name" {
  description = "Public DNS name for the staging ALB."
  value       = aws_lb.app.dns_name
}

output "cluster_name" {
  description = "ECS cluster name."
  value       = aws_ecs_cluster.main.name
}

output "service_name" {
  description = "ECS service name."
  value       = aws_ecs_service.app.name
}

output "ecr_repository_url" {
  description = "ECR repository URL."
  value       = aws_ecr_repository.app.repository_url
}

output "task_definition_family" {
  description = "ECS task definition family."
  value       = aws_ecs_task_definition.app.family
}

output "github_deploy_role_arn" {
  description = "Optional GitHub Actions deploy role ARN."
  value       = try(aws_iam_role.github_deploy[0].arn, null)
}
