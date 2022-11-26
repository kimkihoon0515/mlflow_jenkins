from mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri='http://127.0.0.1:5000')

experiment = client.get_experiment()

print(experiment)