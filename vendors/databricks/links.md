# Databricks — external links

Curated reading list for Databricks reference material.

## Getting started

- [Getting Started with Databricks: A Beginner's Guide](https://medium.com/@mariusz_kujawski/getting-started-with-databricks-a-beginners-guide-8b8db7f6f457) — Mariusz Kujawski. Workspace / Notebooks / Cluster / Metastore primer; cluster types and access modes; Unity Catalog volumes; Delta Lake basics; PySpark reads.

## Unity Catalog & MLflow

- [Lessons learned from migrating models to Unity Catalog](https://medium.com/marvelous-mlops/lessons-learned-from-migrating-models-to-unity-catalog-5cc5b8a973eb) — Marvelous MLOps. 3-catalog (Dev/Staging/Prod) pattern, manual approval via external CI/CD (no UC webhooks), signature enforcement, `search_registered_models()` tag-filter limitations.

## Forecasting

- [A Framework for Multi-Model Forecasting on Databricks](https://www.databricks.com/blog/framework-multi-model-forecasting-databricks) — Databricks Blog. MMF framework: parallel multi-model training on Databricks clusters; standardized `evaluation_output` / `scoring_output` Unity Catalog tables; MLflow + Mosaic AI integration; Model Serving deployment.

## Official docs

- [Databricks Remote Development (SSH tunnel)](https://docs.databricks.com/aws/en/dev-tools/ssh-tunnel)
- [MLOps workflows on Databricks](https://docs.databricks.com/en/machine-learning/mlops/mlops-workflow.html)
- [Manage model lifecycle in Unity Catalog](https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/index.html)
- [AI and machine learning on Databricks](https://docs.databricks.com/aws/en/machine-learning/) — Overview of the integrated platform for building, training, deploying, and managing AI/ML applications across the full lifecycle.
- [What are Unity Catalog volumes?](https://docs.databricks.com/aws/en/volumes) — Unity Catalog objects for governing non-tabular datasets; covers managed vs. external storage types and access control.
- [Databricks architecture](https://docs.databricks.com/aws/en/getting-started/architecture) — Control plane vs. compute plane, lakehouse design patterns (medallion architecture), and ACID guarantees.
- [Connect to serverless compute](https://docs.databricks.com/aws/en/compute/serverless/) — How to configure and use serverless compute for notebooks, workflows, and Lakeflow Spark Declarative Pipelines on AWS.
- [CI/CD on Databricks](https://docs.databricks.com/aws/en/dev-tools/ci-cd/) — CI/CD approaches on Databricks including Declarative Automation Bundles and other pipeline tooling for data engineering and data science.
