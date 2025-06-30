# MLOPs: Week 3 - Orchestration, ML Pipelines
Heiner Atze

# Construction of a Machine Learning pipeline

``` mermaid
graph LR
    one[Download Data] --> two[Data Transformation];
    two --> three[Feature Eng];
    three --> four[Hyperparameter tuning];
    four --> five[Final model fit]
    five --> six(Push to Model registry)
```

## Workflow orchestrators for ML

- Kubeflow pipelines
- MLflow pipelines
