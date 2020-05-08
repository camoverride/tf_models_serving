"""
Hits the server with a request to check if things are working correctly.
Use localhost if testing locally or the base URL or the website if testing online.
"""
import json
import numpy
import requests


url = "34.83.242.28"

# Mimic the shape of the incoming data. First axis are number of images (should always be 1).
image_data = numpy.ones([1, 48, 48, 1])

data = json.dumps({"signature_name": "serving_default", "instances": image_data.tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post(f"http://{url}:8080/v1/models/model:predict", data=data, headers=headers)

predictions = numpy.array(json.loads(json_response.text)["predictions"])

# Should be an array of length 7.
print(predictions[0])
