# tf_models_serving

This repo is where my tensorflow servables live.

## Setup

Take a SavedModel, build it to a docker image, push to GCP, and serve it:

- [Build docs](https://www.tensorflow.org/tfx/serving/docker)
- [Deploy docs](https://www.tensorflow.org/tfx/serving/serving_kubernetes)
- `docker pull tensorflow/serving`
- `docker run -d --name serving_base tensorflow/serving`
<!-- - `docker cp models/emotion_model serving_base:/models/emotion_model`
- `docker commit --change "ENV MODEL_NAME emotion_model" serving_base emotion_model_1` -->
- docker build -t gcr.io/emotion-model-1-276322/emotion-model-1:v0.1 .
- docker push gcr.io/emotion-model-1-276322/emotion-model-1:v0.1
- SERVE IT

## Test Locally

Start the server:

~~~shell
docker run -t --rm -p 8501:8501 \
    -v "/Users/cameronsmith/repos/tf_models_serving/models/emotion_model:/models/emotion_model" \
    -e MODEL_NAME=emotion_model \
    emotion_model_1 &

docker run -t --rm -p 8080:8080 gcr.io/emotion-model-1-276322/emotion-model-1:v0.1 &
~~~

Send a curl request (Python):

~~~python
import json
import numpy
import requests

url = "localhost"
# url = "34.83.242.28"

# Mimic the shape of the incoming data. First axis are number of images (should always be 1).
image_data = numpy.ones([1, 48, 48, 1])

data = json.dumps({"signature_name": "serving_default",
                   "instances": image_data.tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post(f"http://{url}:8080/v1/models/model:predict",
                              data=data, headers=headers)

predictions = numpy.array(json.loads(json_response.text)["predictions"])

print(predictions[0])
~~~
 # 8500 is open to public... 8501 isn't