FROM tensorflow/serving
COPY emotion_model /models/emotion_model/1
COPY gender_model /models/gender_model/1
ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh", "--rest_api_port=8080"]