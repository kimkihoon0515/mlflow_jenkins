model_uri = client.get_model_version_download_uri(name,check_latest_version(name))
        model = bentoml.mlflow.import_model("serving_model",model_uri=f"{model_uri}")
        runner = bentoml.mlflow.get("serving_model:latest").to_runner()

        runners = [runner]

        svc = bentoml.Service("mlflow_demo",runners=[runner])

        input_spec = bentoml.io.NumpyNdarray(
            dtype="float32",
            shape=[-1,784],
            enfore_shape=True,
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
