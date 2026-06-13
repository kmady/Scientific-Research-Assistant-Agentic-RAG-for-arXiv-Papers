# AWS Staging Deployment

This document describes the first production-oriented deployment path for the Agentic RAG project.

## Architecture

```text
GitHub Actions
  -> build Docker image
  -> push to Amazon ECR
  -> update Amazon ECS Fargate service

Application Load Balancer
  -> ECS task on port 8080

ECS task
  -> FastAPI app
  -> Prometheus metrics on port 8000
  -> EFS mounted at /mnt/data
```

## Application Entrypoints

The production container starts:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8080
```

API endpoints:

```text
GET  /health
GET  /ready
POST /query
POST /ingest
```

The existing Prometheus metrics server still runs on `METRICS_PORT`, defaulting to `8000`, when the orchestrator is initialized.

## Terraform

```bash
cd infra/terraform
terraform init
terraform plan
terraform apply
```

Terraform creates:

- VPC, public subnets, internet gateway, and route table
- ECR repository
- ECS cluster, service, task definition, roles, and CloudWatch logs
- ALB, listener, and target group
- EFS file system and access point mounted at `/mnt/data`
- optional GitHub Actions deploy role

By default, the ECS service starts with `desired_count = 0` because the initial `container_image` is only a placeholder. The staging deploy workflow updates the task definition with the real ECR image and then scales the service to one task.

To create the optional GitHub deploy role:

```bash
terraform apply \
  -var='github_oidc_provider_arn=arn:aws:iam::<account-id>:oidc-provider/token.actions.githubusercontent.com' \
  -var='github_repository=<owner>/<repo>'
```

## GitHub Repository Variables

Set these variables in GitHub:

```text
AWS_REGION
AWS_ROLE_TO_ASSUME
ECR_REPOSITORY
ECS_CLUSTER
ECS_SERVICE
ECS_TASK_DEFINITION
ECS_CONTAINER_NAME
```

Typical staging values:

```text
ECR_REPOSITORY=agentic-rag-staging
ECS_CLUSTER=agentic-rag-staging
ECS_SERVICE=agentic-rag-staging
ECS_TASK_DEFINITION=agentic-rag-staging
ECS_CONTAINER_NAME=agentic-rag-api
```

## Secrets

Use AWS Secrets Manager for API keys:

```text
OPENAI_API_KEY
GEMINI_API_KEY
```

Pass them into ECS through Terraform:

```hcl
secret_environment_variables = {
  OPENAI_API_KEY = "arn:aws:secretsmanager:..."
}
```

## Deployment

The `Deploy staging` GitHub Actions workflow:

1. assumes the AWS deploy role through OIDC;
2. logs in to ECR;
3. builds and pushes the Docker image;
4. downloads the current ECS task definition;
5. replaces the app image;
6. deploys the new task definition to the ECS service.

## First Staging Smoke Test

After the first deployment:

```bash
curl http://<alb-dns-name>/health
curl http://<alb-dns-name>/ready
```

Query:

```bash
curl -X POST http://<alb-dns-name>/query \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is the main contribution of paper 2407.14477?","include_steps":false}'
```

For production, add HTTPS with ACM, private subnets with NAT or VPC endpoints, and managed observability through CloudWatch, Amazon Managed Service for Prometheus, and Amazon Managed Grafana.
