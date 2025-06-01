# MLOPs: Week 2 - MLFlow: Experiment tracking
Heiner Atze

# What is experiment tracking ?

Experiment tracking = process of keeping track of all the relevant
infomation from an ML experiment including:

- source code
- model family …

# Why is it important ?

- Reproducibility
- Organization
- Optimization

# MLFlow

- open source platform for the machine learning lifecycle
- Python package with 4 modules:
  - Tracking
  - Models
  - Model Registry
  - Projects

## Tracking experiments in MLflow

Tracking is organized in runs, keeping track of :

- Parameters
- Metrics
- Metadata
- Artifacts (plots)
- Models

Automatic logging of other information of the run:

- source code
- Code version (using `git`)
- start and end time
- author

# Getting started with MLFlow

## Start the MLflow server

``` commandline
mlflow --backend-store-uri sqlite:///mlflow.db
```

## Experiment tracking from within python

``` python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

...

with mlflow.start_run():
    mlflow.set_tag("developer", "me")

    mlflow.log_param('train_data_path', '../week1/train_model/data/green_tripdata_2021-01.parquet')
    mlflow.log_param('valid_data_path', '../week1/train_model/data/green_tripdata_2021-02.parquet')

    alpha = 0.01
    mlflow.log_param("alpha", alpha)
    lr = Lasso(alpha)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)
    
    mlflow.log_metric("rmse", rmse)
```

# Tracking a hyperparameter search

## Custom objective in `hyperopt`

``` python
def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

    return {'loss': rmse, 'status': STATUS_OK}
```

``` python
search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:linear',
    'seed': 42
}

best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=Trials()
)
```

## Automatic logging for supported modeling frameworks

https://mlflow.org/docs/latest/tracking/autolog

``` python
mlflow.xgboost.autolog()

booster = xgboost.train(
  params = params
  ...
)
```

# Model Management

Comprises:

1.  Experiment tracking
2.  Model Versioning
3.  Model Deployment

## Model Versioning

### Artifacts

The simplest possibility is to save the pickle model as an artifact.

``` python
...
mlflow.log_artifact(local_path = './path/to/model', artifact_path = './path/to/artifact')
```

### Autologging of models and pipelines

``` python
...
# equivalent to output from autologging
mlflow.xgboost.log_model(model, artifact_path = './path/to/artifact')
...
```

- pipeline steps can be logged with `mlflow.log_artifact`

- logged model can accessed using its `uri`

``` python
#loaded_model = python function
loaded_model = mlflow.pyfunc.load_model(logged_model_uri)
#loaded_model = object
loaded_model = mlflow.<framework>.load_model(logged_model_uri)
```

# Model registry

The MLflow Model Registry is a centralized repository for managing the
lifecycle of MLflow Models. It provides model lineage, versioning, stage
transitions (e.g., staging, production, archived), and annotations.

**Key Features:**

- **Model Management:** Register, version, and manage MLflow models.
- **Model Lineage:** Track the experiment runs that produced the models.
- **Stages:** Assign stages (e.g., “Staging”, “Production”) to models.
- **Transitions:** Transition models between stages.
- **Annotations:** Add descriptions, tags, and comments.

## Register models

### Using `mlflow` run

``` python
import mlflow
import mlflow.xgboost

# Assuming you have already trained a model using autologging
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Replace with your actual URI if different

# Start an MLflow run
with mlflow.start_run():
    # Autologging will log the model automatically
    mlflow.xgboost.autolog()

    # Your model training code here (example)
    params = {
        'max_depth': 10,
        'learning_rate': 0.1,
        'n_estimators': 100
    }

    # Train your model (assuming 'train' is your training dataset)
    booster = xgboost.train(
        params=params,
        dtrain=train,
        num_boost_round=1000,
        evals=[(valid, 'validation')],
        early_stopping_rounds=50
    )

    # Register the model to the model registry
    # Version=1 if not existing
    # Verison += 1 if model exists already
    mlflow.xgboost.log_model(
        "model", artifact_path="model"
    )
```

### Using the registry API

``` python
# Assuming you have already trained and logged a model named 'my_model'
# Specify the MLflow tracking URI (where your model registry is stored)
mlflow_tracking_uri = "sqlite:///mlflow.db"  # Replace with your actual URI if different
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Log the model to the model registry
model_uri = "runs:/{}/model".format(run_id)  # Replace run_id with the actual run id where the model was logged
model_name = "my_model_name"

# Register the model
mlflow.register_model(model_uri=model_uri, name=model_name)

print(f"Model registered: {model_name} at {model_uri}")
```

``` python
# Search for runs
experiment_name = "nyc-taxi-experiment" # Replace with your experiment name
runs = mlflow.search_runs(experiment_names=[experiment_name])

# Print run information
if runs.empty:
    print(f"No runs found for experiment '{experiment_name}'.")
else:
    print(f"Found {len(runs)} runs for experiment '{experiment_name}':")
    for run in runs:
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Status: {run.info.status}")
        print(f"  Start Time: {run.info.start_time}")
        print(f"  End Time: {run.info.end_time}")
        print(f"  Parameters: {run.data.params}")
        print(f"  Metrics: {run.data.metrics}")
        print("-" * 30)
```

## Promoting a Model

Models can be promoted from one stage to another (e.g., from Staging to
Production) through the MLflow UI or programmatically. This allows you
to test and validate models before deploying them for live use.

``` python
# Importing the MLflow client
import mlflow

# Assuming you have already trained and logged a model named 'my_model'
# and it's registered in the MLflow Model Registry with the name 'my_model_name'

# Specify the MLflow tracking URI (where your model registry is stored)
mlflow_tracking_uri = "sqlite:///mlflow.db"  # Replace with your actual URI if different

# Get the latest version of the model in the 'Staging' stage
client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_tracking_uri)
model_name = "my_model_name"
latest_version = client.get_latest_versions(model_name, stages=["Staging"])[0].version

# Transition the model to the 'Production' stage
client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage="Production"
)


# change description
client.update_model_version(
    name = model_name,
    version = latest_version,
    description = f"Model '{model_name}' version {latest_version} transitioned to 'Production'."
)
```

# MLflow in practice

Scenarios:

1.  Solo Data Scientist in ML competition

    `mlflow` remote server = overkill model registry not necessary

2.  Cross-functional team, single Data Scientist working on an ML model

    local `mlflow` server might be sufficient registry is probably
    useful

3.  Multiple data scientitst working on multiple models

    remote `mlflow` is probably necessary sharing information is crucial
    for collaboration

## Configuring MLflow

- Backend store (local FS? SQLAlchemy compatible DB)
- Artifacts store (local FS or remote: S3 buckets ?)
- Tracking server (none, local, remote ?)

## Scenario showcases

Notebooks at [Zoomcamp github
repo](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/02-experiment-tracking/running-mlflow-examples)

### Scenario 1

- local `mlflow` usage, no server

### Scenario 2

- using a local `mlflow` server
- backend using `sqlite`

``` bash
mlflow sever --backend-store-uri sqlite:///backend.db --default-artifact-root ./artifacts_local
```

- set tracking uri in python to the server address

``` python
import mlflow

mlflow.set_tracking_uri('http://127.0.0.1:5000')
```

### Scenario 3

- remote `mlflow` server is necessary
- backend store : remote `postgres`
- artifcats using S3 bucket

**Benefits** of a remote tracking server

- easy sharing, collaboration and model visibility

**Issues to address**

- security concerns need to be addressed
- scalability
- isolation (naming conventions for experiments, models, default tags)
- restriction of artifact access (separate artifact stores per
  experiments)

**Limitations**

- open-source version provides no auth
- Data versioning is not included
- Model or Data monitoring and alerting in production not in scope of
  `mlflow`

**Alternatives**

- Neptune
- Comet
- Weights & Biases …
