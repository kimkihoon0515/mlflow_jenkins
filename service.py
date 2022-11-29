import bentoml
import numpy as np
from mlflow.tracking import MlflowClient


client = MlflowClient(tracking_uri="http://13.125.54.121:5000")
from check_version import check_latest_version

name = 'basic_model'

model_uri = client.get_model_version_download_uri(name,check_latest_version(name))
print(model_uri)
bento_model = bentoml.mlflow.import_model("serving_model","s3://sojt/1/0c6027c2bd934cc3bffd1eab98397b3c/artifacts/best_model/data/model.pth")
