# tf_models_serving

This repo is where my tensorflow servables live.

## Overview

Take a SavedModel, build it to a docker image, push to GCP, and serve it.

Models live in the `models` folder, and should all be in the format of `models/<model_name>/<model_version>`.

Make sure that models are created with the `serving_default` signature, or they won't be able to be served.

- [Build docs](https://www.tensorflow.org/tfx/serving/docker)
- [Deploy docs](https://www.tensorflow.org/tfx/serving/serving_kubernetes)

## Use this Repo

Build an image: `docker build -t camoverride/face-models:v0.1 .`

Run it: `docker run -t --rm -p 8080:8080 camoverride/face-models:v0.1 --model_config_file=/models/models.config &`

Not: port `8500` is open to public... `8501` isn't.

## Test Locally

Send a curl request (Python):

~~~python
import json
import numpy
import requests

url = "localhost"
port = "8080"
model = "gender_model"
# model = "emotion_model"

# Mimic the shape of the incoming data. First axis are number of images.
image_data = numpy.random.rand(3, 224, 224, 3) # for gender_model
# image_data = numpy.random.rand(3, 48, 48, 1) # for emotion_model

data = json.dumps({"signature_name": "serving_default",
                   "instances": image_data.tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post(f"http://{url}:{port}/v1/models/{model}:predict", data=data, headers=headers)

predictions = numpy.array(json.loads(json_response.text)["predictions"])

print(predictions)
~~~

## To-do list

- Models accept differently-sized inputs -- these need to be standardized.
- The model server is not taking advantage of GPU, so it might be a little slow.
