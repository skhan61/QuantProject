#!/bin/bash

# Read credentials from file
read -r DOCKER_USER < docker_creds.txt
read -r DOCKER_PASS

# Login to Docker Hub
echo "$DOCKER_PASS" | sudo docker login -u "$DOCKER_USER" --password-stdin

# Tag your local image with the desired tag
sudo docker tag myproject:latest k1103061/quantproject_v01:v2023.09

# Push the newly tagged image to Docker Hub
sudo docker push k1103061/quantproject_v01:v2023.09
