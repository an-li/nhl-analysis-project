#!/bin/bash

docker build --tag serving . -f Dockerfile.serving
docker build --tag streamlit . -f Dockerfile.streamlit