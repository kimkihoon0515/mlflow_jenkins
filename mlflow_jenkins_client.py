from mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri='http://127.0.0.1:5000')
from mlflow_train import name

filter_string = 'basic_model'

results = client.search_model_versions(filter_string)

for res in results:
    print("run_id={}; version={}; current_stage={}".format(res.run_id,res.version,res.current_stage))