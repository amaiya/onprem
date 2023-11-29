#!/usr/bin/env bash

if [ "$EUID" -ne 0 ]; then
    echo "This script must be run with sudo."
    exit 1
fi

function print_guide() {
    echo "Usage: $0 [CUDA version]"
    echo "  where CUDA version may be 11 or 12. 12 is used if omitted"
    exit 1
}

# Are we building CUDA 12 or CUDA 11?
cuda_version=12
if [ "$#" -eq 0 ]; then
    echo "CUDA version not provided. Using default of ${cuda_version}"
elif [ "$#" -eq 1 ]; then
    if [ "$1" -ne 11 ] && [ "$1" -ne 12 ]; then
        print_guide
    fi
    cuda_version=$1
else
  print_guide
fi


suffix="12.1.1"
if [ "$cuda_version" -eq 11 ]; then
    suffix="11.6.1"
fi


# Get the directory of the script
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script's directory
cd "$script_dir"

docker build -t cuda_simple:$suffix -f Dockerfile-cuda_simple-$suffix .
docker build -t onprem_cuda:$suffix -f Dockerfile-$suffix ..
