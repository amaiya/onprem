#!/usr/bin/env bash

if [ "$EUID" -ne 0 ]; then
    echo "This script must be run with sudo."
    exit 1
fi

# Get the directory of the script
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script's directory
cd "$script_dir"

docker build -t cuda_simple -f Dockerfile-cuda_simple .
docker build -t onprem_cuda .
