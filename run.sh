#!/bin/bash

docker run -e COMET_API_KEY=$COMET_API_KEY -p 8080:8080 serving
docker run -e SERVING_IP="serving" -e SERVING_PORT="8080" -p 8081:8081/tcp -p 8081:8081 streamlit