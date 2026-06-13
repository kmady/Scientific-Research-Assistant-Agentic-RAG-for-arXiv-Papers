variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Base name for AWS resources."
  type        = string
  default     = "agentic-rag"
}

variable "environment" {
  description = "Deployment environment name."
  type        = string
  default     = "staging"
}

variable "container_image" {
  description = "Initial container image. CI/CD replaces this after the first push to ECR."
  type        = string
  default     = "public.ecr.aws/docker/library/python:3.12-slim"
}

variable "desired_count" {
  description = "Desired ECS service task count."
  type        = number
  default     = 0
}

variable "app_cpu" {
  description = "Fargate task CPU units."
  type        = number
  default     = 2048
}

variable "app_memory" {
  description = "Fargate task memory in MiB."
  type        = number
  default     = 8192
}

variable "llm_provider" {
  description = "LLM provider for staging."
  type        = string
  default     = "mock"
}

variable "embedding_provider" {
  description = "Embedding provider for staging."
  type        = string
  default     = "local"
}

variable "secret_environment_variables" {
  description = "Map of environment variable names to Secrets Manager ARNs."
  type        = map(string)
  default     = {}
}

variable "github_oidc_provider_arn" {
  description = "Existing IAM OIDC provider ARN for token.actions.githubusercontent.com. Leave empty to skip deploy role creation."
  type        = string
  default     = ""
}

variable "github_repository" {
  description = "GitHub repository allowed to deploy, in owner/name format. Leave empty to skip deploy role creation."
  type        = string
  default     = ""
}

variable "github_deploy_branch" {
  description = "Git branch allowed to assume the deploy role."
  type        = string
  default     = "main"
}
