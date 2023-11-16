Steps to Install onprem with GPU on WSL2
========================================

I tested this on two home systems, a custom-built rig with a GeForce RTX 3060
(12 GiB VRAM), and the other a laptop with a GeForce RTX 3050 Ti (4 GiB VRAM).
Both systems had Windows 11 installed.

1. Make sure latest Windows device drivers are installed. I used
   [Nvidia GeForce Experience](https://www.nvidia.com/en-us/geforce/geforce-experience/)
   to accomplish this.
2. Install Ubuntu 22.04 instance in WSL: `wsl --install -d Ubuntu-22.04`
3. Check `nvidia-smi` output in Ubuntu. Verify that it shows a device with Nvidia
   driver supporting CUDA 12.3.
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
