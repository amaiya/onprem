#!/usr/bin/env bash

function print_guide() {
    echo "Usage: $0"
    echo
    echo "NVIDIA CUDA support version will be automatically detected. Only"
    echo "versions 11 and 12 are supported."

}

# Only allow script to run as root
if [ "$EUID" -ne 0 ]
then
    echo "This script must be run with sudo."
    exit 1
fi

# Get the directory of the script, and change to it
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$script_dir"

function build_cpu_image() {
    echo "Building the CPU-only image, called onprem:cpu."
    docker build -t onprem:cpu -f Dockerfile-cpu ..
}

# Is the necessary NVIDIA stuff installed? Technically, we could build the image
# without, but it is assumed that the container image will be used locally.
if ! command -v nvidia-smi >/dev/null 2>&1
then
  echo "The nvidia-smi executable is not present. Are you sure you have an"
  echo "NVIDIA device with NVIDIA drivers installed? Don't worry! I'll still"
  echo "build you a CPU-only image."
  build_cpu_image
  exit
fi

# Get device CUDA version, and find a supporting image tag, and Torch package index
cuda_version=`nvidia-smi | grep -P -o "CUDA Version: \d+(\.\d+)+" | grep -P -o "\d+(\.\d+)+"`
echo "Detected driver supporting CUDA Version ${cuda_version}"
tag=`./get_cuda_image_tag ${cuda_version}`
cuda_image=${tag}-devel-ubuntu22.04
case "$cuda_image" in
  "12"*)
    torch_index_url=https://pypi.org/pypi
    ;;
  "11.8"*)
    torch_index_url=https://download.pytorch.org/whl/cu118
    ;;
  "11.7"*)
    torch_index_url=https://download.pytorch.org/whl/cu117
    ;;
  *)
    echo "I don't know which package index to use for PyTorch with $cuda_image"
    build_cpu_image
    exit
    ;;
esac

echo "Building onprem_cuda:$tag based on nvidia/cuda:$cuda_image and using"
echo "$torch_index_url for the torch packages"
docker build --build-arg CUDA_IMAGE=$cuda_image \
  --build-arg TORCH_INDEX_URL=$torch_index_url -t onprem_cuda:$tag \
  -f Dockerfile-cuda ..

build_cpu_image
