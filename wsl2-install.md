Steps to Install onprem with GPU on WSL2
========================================

1. `wsl --install -d Ubuntu-22.04`
2. Check nvida-smi output. On my (son's) system, it shows a GeForce RTX 3060
   with driver supporting CUDA 12.3 support.
3. Do all [steps](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)
   in the Official CUDA 12.3 toolkit install for WSL v2.
   (I saw a warning: */usr/lib/wsl/lib/libcuda.so.1* is not a symbolic link)
4. `apt install python3.10-dev python3.10-venv build-essential`
5. Make your venv and activate it.
6. `apt install nvidia-cuda-toolkit`
7. Install llama-cpp-python with cuBLAS support:
   `CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python`
8. `pip install onprem`
9. Run an onprem prompt with `n_gpu_layers=35`. Got an error: "No cuda-capable
   device detected!" Even after shutting down VM and restarting it, still saw
   error.
10. Hail Mary: Let’s try installing
    [lambda stack](https://lambdalabs.com/lambda-stack-deep-learning-software).
    I still have the same problem.
11. Created another venv that inherits the system site packages, then repeated
    steps 8-10. It works!

The question remains. Did I really need to install the lambda stack? I didn't
snapshot the system before doing that, so I will have to repeat steps 1-9,
making sure in step 5 to inherit system site packages.
