FROM tensorflow/serving
COPY emotion_model /models/model/1
ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh", "--rest_api_port=8080"]