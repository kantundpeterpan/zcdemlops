import pickle
from flask import Flask, request, jsonify

RUN_ID = 'ecf5c933c8d04c0cb32ac726b561ed13'
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'

import os
# os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI

import mlflow
from mlflow.tracking import MlflowClient

# client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
# logged_model = 'mlflow-artifacts:/2/models/m-dbbe1bfd8e8c44e88f459ca9df82f691/artifacts'
logged_model = '../../mlartifacts/2/models/m-dbbe1bfd8e8c44e88f459ca9df82f691/artifacts'
# Load model as a PyFuncModel.
model = mlflow.pyfunc.load_model(logged_model)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features
    
def predict(features):
    # X = dv.transform(features)
    preds = model.predict(features)
    
    return preds[0]

app = Flask("duration-prediction")

@app.route('/predict', methods = ['POST'])
def predict_endpoint():
    ride = request.get_json()
    
    
    features = prepare_features(ride)
    pred = predict(features)
    
    result = {
        'duration': pred
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host = '0.0.0.0', port=9696)