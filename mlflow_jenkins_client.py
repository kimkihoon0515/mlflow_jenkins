from mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri='http://127.0.0.1:5001')
import requests
import bentoml
import numpy as np
import json


def production_check(name):
    datas = {
        'name': name,
        'stages': 'Production'
    }
    url = "http://localhost:5001/api/2.0/mlflow/registered-models/get-latest-versions"

    response = requests.get(url,params=datas).json()

    try:
        version = response['model_versions'][-1]['version']
        run_id = response['model_versions'][-1]['run_id']
        return(version,run_id)
    except:
        return('There are no serving models')

name = 'basic_model'

version = production_check(name)[0]

def get_model_uri(name,version):
    datas = {
        'name': name,
        'version': version
    }

    url = "http://localhost:5001/api/2.0/mlflow/model-versions/get-download-uri"

    response = requests.get(url,params=datas).json()

    model_uri = response['artifact_uri']

    return model_uri

model_uri = get_model_uri(name,version)

model = bentoml.mlflow.import_model("serving_model",model_uri=f"{model_uri}")

print(f"Model imported to BentoML: {model}" )

with open('./test_input.json', 'r') as f:
  test_input_arr = np.array(json.load(f), dtype=np.float32)

print(len(test_input_arr))

runner = bentoml.mlflow.get("serving_model:latest").to_runner()

runners = [runner]

svc = bentoml.Service("mlflow_demo",runners=[runner])

input_spec = bentoml.io.NumpyNdarray(
    dtype="float32",
    shape=[-1,784],
    enforce_shape=True,
    enforce_dtype=True,
)

@svc.api(input=input_spec,output=bentoml.io.NumpyNdarray())
def predict(input_arr):
    result = runner.predict.run(input_arr)

    answer = []
    for i in result:
        answer.append(np.argmax(i))

    answer = np.array(answer)

    return answer