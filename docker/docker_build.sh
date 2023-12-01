#!/usr/bin/env bash

function print_guide() {
    echo "Usage: $0"
    echo
    echo "NVIDIA CUDA support version will be automatically detected. Only"
    echo "versions 11 and 12 are supported."
    exit 1
}

# Only allow script to run as root
if [ "$EUID" -ne 0 ]
then
    echo "This script must be run with sudo."
    exit 1
fi

# Is the necessary NVIDIA stuff installed? Technically, we could build the image
# without, but it is assumed that the container image will be used locally.
if ! command -v nvidia-smi >/dev/null 2>&1
then
  echo "The nvidia-smi executable is not present. Are you sure you have an"
  echo "NVIDIA device with NVIDIA drivers installed?"
  exit 1
fi

# Are we building CUDA 12 or CUDA 11?
cuda_version=`nvidia-smi | grep -P -o "CUDA Version: \d+(\.\d+)*" | grep -P -o "\d+" | head -n 1`
echo "Detected driver supporting CUDA ${cuda_version}."
if [ "$cuda_version" -ne 11 ] && [ "$cuda_vesion" -ne 12 ]
then
    print_guide
fi

# Set the suffix of the Dockerfile to use
suffix="12.1.1"
if [ "$cuda_version" -eq 11 ]; then
    suffix="11.6.1"
fi
echo "Building the CUDA ${suffix} container images."

# Get the directory of the script
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script's directory
cd "$script_dir"

docker build -t cuda_simple:$suffix -f Dockerfile-cuda_simple-$suffix ..
docker build -t onprem_cuda:$suffix -f Dockerfile-$suffix ..
