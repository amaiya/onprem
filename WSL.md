How to Run OnPrem On WSL
========================

This has been on two Windows Home 11 systems:

1. A custom-built minitower PC with ASUS motherboard a GeForce RTX 3060 (12 GiB
   VRAM)
2. An HP Victus laptop PC with a GeForce RTX 3050 Ti (4 GiB VRAM)

First, Install Ubuntu 22.04 WSL Instance
----------------------------------------

This is quite simple to do. At a command prompt: `wsl --install -d Ubuntu-22.04`

Ubuntu 22.04 is a preference of the author, not a requirement. Almost any instance
type with a modern enough Python version will work just fine. However, the
NVIDIA GPU section below has only been tested on Ubuntu 22.04 instances.

Once the environment started for the first time, WSL works just like a native
Ubuntu installation. If you have an NVIDIA device supporting CUDA, you can even
use it to run models faster.

Setting Up CUDA for Onprem on Your WSL Instance
-----------------------------------------------

1. Make sure latest Windows device drivers are installed. I used
   [Nvidia GeForce Experience](https://www.nvidia.com/en-us/geforce/geforce-experience/)
   to accomplish this.
2. Check `nvidia-smi` output in Ubuntu. Verify that it shows a device with Nvidia
   driver supporting CUDA 12.3. You can skip the rest of the step and follow the
   [Docker instructions](#how-to-run-onprem-in-a-docker-container) if you intend to
   use this WSL 2 environment as a Docker host for *onprem* containers.
3. `apt install build-essential`
4. Do all [runfile (local) installation](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local)
   of the Official CUDA 12.3 toolkit install for WSL v2. Ensure the following
   warnings from the final summary output get addressed (If not done, you may
   see a "No CUDA-capable device detected" when attempting to run onprem with
   GPU layers.):
   * PATH includes /usr/local/cuda-12.3/bin
   * LD_LIBRARY_PATH includes /usr/local/cuda-12.3/lib64, or, add
     /usr/local/cuda-12.3/lib64 to /etc/ld.so.conf and run `ldconfig` as root
5. `apt install python3.10-dev python3.10-venv`
6. Make your venv using the python 3.10 executable (should be `python3` on your
   *PATH*). I used the `--system-site-packages` option, though this may not have
   been necessary.
7. Activate your new venv.
8. Install llama-cpp-python with cuBLAS support:
   `CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python`
9. `pip install onprem`
10. Run an onprem prompt with `n_gpu_layers=35`. This may be too much for the 4
    GiB GPU. I found `n_gpu_layers=31` fits in memory, and didn't go any higher.