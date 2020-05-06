import os

# from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if physical_devices:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

file = tf.keras.utils.get_file(
    "grace_hopper.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
# plt.imshow(img)
# plt.axis('off')
x = tf.keras.preprocessing.image.img_to_array(img)
x = tf.keras.applications.mobilenet.preprocess_input(
    x[tf.newaxis,...])

# print(x)

import json
import numpy
import requests

data = json.dumps({"signature_name": "serving_default",
                        "instances": x.tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/emotion_model:predict',
                                    data=data, headers=headers)
print(json_response.text)

print(predictions)
# labels_path = tf.keras.utils.get_file(
#     'ImageNetLabels.txt',
#     'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
# imagenet_labels = np.array(open(labels_path).read().splitlines())

# pretrained_model = tf.keras.applications.MobileNet()
# result_before_save = pretrained_model(x)

# decoded = imagenet_labels[np.argsort(result_before_save)[0,::-1][:5]+1]

# print("Result before saving:\n", decoded)

# mobilenet_save_path = "./mobilenet/1/"
# tf.saved_model.save(pretrained_model, mobilenet_save_path)
