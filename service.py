import bentoml
import numpy as np
from mlflow.tracking import MlflowClient
import os

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "True"


client = MlflowClient(tracking_uri="http://13.125.54.121:5000")
from check_version import check_latest_version

name = 'basic_model'

model_uri = client.get_model_version_download_uri(name,check_latest_version(name))
print(model_uri)
bento_model = bentoml.mlflow.import_model("serving_model",model_uri)

input_spec = bentoml.io.NumpyNdarray(
dtype="float32",
shape=[-1,784],
enforce_shape=True,
enforce_dtype=True
)

runner = bentoml.mlflow.get("serving_model:latest").to_runner()

runners = [runner]

svc = bentoml.Service("mlflow_demo",runners=[runner])

@svc.api(input=input_spec,output=bentoml.io.NumpyNdarray())
def predict(input_arr):
    result = runner.predict.run(input_arr)

    answer = []

    for i in result:
        answer.append(np.argmax(i))

    answer = np.array(answer)

    return answer