Onprem README for Running in WSL 2 or With Docker
=================================================

How to Install onprem with GPU on WSL2
--------------------------------------

I tested this on two home systems, a custom-built rig with a GeForce RTX 3060
(12 GiB VRAM), and the other a laptop with a GeForce RTX 3050 Ti (4 GiB VRAM).
Both systems had Windows 11 installed.

1. Make sure latest Windows device drivers are installed. I used
   [Nvidia GeForce Experience](https://www.nvidia.com/en-us/geforce/geforce-experience/)
   to accomplish this.
2. Install Ubuntu 22.04 instance in WSL: `wsl --install -d Ubuntu-22.04`
3. Check `nvidia-smi` output in Ubuntu. Verify that it shows a device with Nvidia
   driver supporting CUDA 12.3. You can jump down to the
   [next section](#how-to-run-onprem-in-a-docker-container) if you intend to
   use this WSL 2 environment as a Docker host for *onprem* containers.
4. `apt install build-essential`
5. Do all [runfile (local) installation](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local)
   of the Official CUDA 12.3 toolkit install for WSL v2. Ensure the following
   warnings from the final summary output get addressed (If not done, you may
   see a "No CUDA-capable device detected" when attempting to run onprem with
   GPU layers.):
   * PATH includes /usr/local/cuda-12.3/bin
   * LD_LIBRARY_PATH includes /usr/local/cuda-12.3/lib64, or, add
     /usr/local/cuda-12.3/lib64 to /etc/ld.so.conf and run `ldconfig` as root
6. `apt install python3.10-dev python3.10-venv`
7. Make your venv and activate it. I used the `--system-site-packages` option,
   though this may have not been necessary.
8. Install llama-cpp-python with cuBLAS support:
   `CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python`
9. `pip install onprem`
10. Run an onprem prompt with `n_gpu_layers=35`. This may be too much for the 4
    GiB GPU. I found `n_gpu_layers=31` fits in memory, and didn't go higher.

How to Run onprem in a Docker Container
---------------------------------------

1. Confirm that your host has an Nvidia device and driver installed, supporting
   CUDA. On a Linux host, the `nvidia-smi` command-line utility can verify this.
   On a Windows host, launch *NVIDIA Control Panel*, and click on
   *System Information* in the lower-left-hand side of its window.
2. Install Docker. On Ubuntu Linux 22.04 (or WSL 2 running it), `apt install docker.io`
   will accomplish this.
3. Install the NVIDIA Container Toolkit. Full instructions, including necessary
   configuration steps, are at
   [this link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
4. Test that the NVIDIA Container Toolkit is passing through the NVIDIA Driver
   to properly run containers by following the post-installation instructions
   titled *Running a Sample Workload*.
5. Build the *onprem_cuda* image by running `docker/docker_build.sh`
6. The following shell command will launch the image's python interpreter in a
   container, mapping the host's persistent *~/onprem_data/* folder to the
   correct location inside the container.

   ```shell
   $ sudo docker run --rm -it --gpus=all --cap-add SYS_RESOURCE -e USE_MLOCK=0 \
         -v ~/onprem_data:/home/root/onprem_data cuda_simple
   >>> from onprem import LLM
   >>> llm = LLM(n_gpu_layers=35)
   … et cetera …
   ```

If you wish to run the Streamlit application from the container, this command
will launch it on a given port, mapping that port to be accessible from the
host:

```shell
$ sudo docker run --gpus=all --cap-add SYS_RESOURCE -e USE_MLOCK=0 \
    -v ~/onprem_data:/home/root/onprem_data -p 8000:8000 cuda_simple \
    onprem --port 8000
```
