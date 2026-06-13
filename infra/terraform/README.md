# AWS Staging Infrastructure

Minimal Terraform foundation for staging:

- ECR repository
- ECS Fargate cluster, task definition, and service
- Application Load Balancer on HTTP port 80
- CloudWatch log group
- EFS file system mounted at `/mnt/data`

## Usage

```bash
cd infra/terraform
terraform init
terraform plan
terraform apply
```

The first apply uses `container_image` as a placeholder and creates the ECS service with `desired_count = 0`. After ECR exists, GitHub Actions can build and deploy the real image, then scale staging to one task.

To create the optional GitHub Actions deploy role, provide:

```bash
terraform apply \
  -var='github_oidc_provider_arn=arn:aws:iam::<account-id>:oidc-provider/token.actions.githubusercontent.com' \
  -var='github_repository=<owner>/<repo>'
```

If the IAM OIDC provider does not exist yet, create it once in the AWS account for:

```text
https://token.actions.githubusercontent.com
```

## Required GitHub Variables

Set these repository variables for deploy workflow:

```text
AWS_REGION
AWS_ROLE_TO_ASSUME
ECR_REPOSITORY
ECS_CLUSTER
ECS_SERVICE
ECS_TASK_DEFINITION
ECS_CONTAINER_NAME
```

Typical values after Terraform:

```text
ECR_REPOSITORY=agentic-rag-staging
ECS_CLUSTER=agentic-rag-staging
ECS_SERVICE=agentic-rag-staging
ECS_TASK_DEFINITION=agentic-rag-staging
ECS_CONTAINER_NAME=agentic-rag-api
AWS_ROLE_TO_ASSUME=<github_deploy_role_arn output>
```
