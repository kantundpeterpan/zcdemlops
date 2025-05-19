# MLOPs: Week 1 - Introduction
Heiner Atze

# Stages of an ML project

## Design

- Do we need ML ?

## Train

- Model training and selection

## Operate

MLOps = set of best practices to ensures smooth operations of a deployed
ML model

# Optional: Predict ride duration

see [notebook](./train_model/duration_predicion.ipynb)

# Breaking up the notebook and take it further

- keep track of models : `MLFlow`
- pack repetitive steps into `Pipelines`
- Deployment
- Fully automated CI/CD

# MLOps Maturity Model

- Levels 0 (no MLOps at all) to Level 4 (fully automated)

Ref. [Machine Learning operations maturity
model](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model)

## 0 - No MLOps

- Notebook based
- no pipelines, model tracking nor metadata
- no automation
- fine for POC projects

## 1 - DevOps, No MLOps

- some level of automation for releases
- unit integration tests
- CI/CD
- OPS metrics

NO - experiment tracking - reproducibility - DS still separated from
Engineers

## 2 - Automated training

- does not rely on Notebooks
- experiment tracking
- model registry
- low friction deployment
- DS works with Eng.

## 3 - Automated deployment

- Easy to deploy model

- e.g. API calls to ML platform to deploy

- deployment can be part of the training pipeline

- A/B tests can be integrated in the ML platform

- Model monitoring

## 4 - Full MLOps Automation

# Homework

see
[notebook](https://nbviewer.org/github/kantundpeterpan/zcdemlops/blob/main/week1/train_model/duration_prediction_yellow.ipynb)
