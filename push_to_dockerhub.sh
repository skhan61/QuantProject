#!/bin/bash

# Exit on any non-zero status.
set -e

# 1. Build the Docker container
sudo docker build -t myproject:latest .

# Check the image size, and ensure it's less than 5GB for any layer.
image_size=$(sudo docker inspect myproject:latest --format='{{.Size}}')
if (( $image_size > 5000000000 )); then
    echo "Image size is greater than 5GB. Exiting."
    exit 1
fi

# Print the image size in a human-readable format
echo "Image size: $(numfmt --to=iec-i --suffix=B $image_size)"

# 2. Get the current date in YYYY-MM-DD format
current_date=$(date +"%Y-%m-%d")

exec 3<docker_creds.txt

# Read credentials from file
read -r DOCKER_USER <&3
read -r DOCKER_PASS <&3
exec 3<&-


# 3. Login to Docker Hub
echo "$DOCKER_PASS" | sudo docker login -u "$DOCKER_USER" --password-stdin

# 4. Tag your local image with the desired tag using the current date
tagged_image_name="k1103061/quantproject_v01:$current_date"
sudo docker tag myproject:latest $tagged_image_name

# 5. Push the newly tagged image to Docker Hub
echo "Pushing the image to Docker Hub..."
sudo docker push $tagged_image_name

# 6. Remove the local images to save space after pushing
sudo docker image rm myproject:latest $tagged_image_name
