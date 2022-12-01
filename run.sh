#!/bin/bash

docker run -e COMET_API_KEY=$COMET_API_KEY -p 8080:8080/tcp -p 8080:8080/udp serving