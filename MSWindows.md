# How to Install and Run OnPrem.LLM on Microsoft Windows

When using OnPrem.LLM on Microsoft Windows (e.g., Windows 11), Windows Subsystem for Linux (WSL2) is recommended over using Windows directly. However, this guide provides instructions for both.


## Using System Python in Windows 11

1. Download and install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and make sure **Desktop development with C++** workload is selected and installed. This is needed to build `chroma-hnswlib` (as of this writing a pre-built wheel only exists for Python 3.11 and below). It is also needed if you need to build `llama-cpp-python` instead of installing a prebuilt wheel (as we do below).
2. Install Python 3.12:  Open "cmd" as administrator and type python to trigger *Microsoft Store* installation in Windows 11.
3. Create virtual environment: `python -m venv .venv`
4. Activate virtual environment: `.venv\Scripts\activate`. You can optionally append `C:\Users\<username\.venv\Scripts` to `Path` environment variable, so that you only need to type `activate` to enter virtual environment in the future.
5. Install PyTorch:
   - For CPU: `pip install torch torchvision torchaudio`
   - For GPU (if you do happen to have a GPU and have installed an NVIDIA driver ): `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
     - Run the following at Python prompt to verify things are working. (The PyTorch binaries ship with all CUDA runtime dependencies and you don't need to locally install a CUDA toolkit or cuDNN, as long as NVIDIA driver is installed.)
	   ```python
	   In [1]: import torch
	
	   In [2]: torch.cuda.is_available()
	   Out[2]: True
	
	   In [3]: torch.cuda.get_device_name()
	   Out[3]: 'NVIDIA RTX A1000 6GB Laptop GPU'
	   ```
7. Install **llama-cpp-python** (CPU only) using pre-built wheel: `pip install llama-cpp-python==0.2.90 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`
  - If you want the LLM to generate faster answers using your GPU, then you'll need to install the CUDA Toolkit, as described in [this guide](https://medium.com/@piyushbatra1999/installing-llama-cpp-python-with-nvidia-gpu-acceleration-on-windows-a-short-guide-0dfac475002d). You will also need to build `llama-cpp-python` with CUDA support, as described [here](https://python.langchain.com/docs/integrations/llms/llamacpp/#installation-with-windows).
8. Install OnPrem.LLM: `pip install onprem `
9. [OPTIONAL] If you're behind a corporate firewall and  have SSL certificate issues, you can try adding `REQUESTS_BUNDLE` as an environment variable and point it to certs for your organization if behind a corporate, so hugging face models can be downloaded. Without this steup, you will need to use the `--trusted-host` option
10. [OPTIONAL] Enable long paths if you get an error indicating you do:  https://stackoverflow.com/questions/72352528/how-to-fix-winerror-206-the-filename-or-extension-is-too-long-error/76452218#76452218
11. Try onprem at a Python prompt to make sure it works. Run the `python` command and type the following:
     ```python
     from onprem import LLM
     llm = LLM()
     llm.prompt('List three cute names for a cat.')

     # On a multi-core 2.5 GHz laptop CPU (e.g., *13th Gen Intel(R) Core(TM) i7-13800H 2.50 GHz*), you should get speeds of around 12 tokens per second.
     ```

## Using WSL2 (with GPU Acceleration)

(These steps for WSL2 were contributed by @dvisser.)


This has been on two Windows Home 11 systems:

1. A custom-built minitower PC with ASUS motherboard a GeForce RTX 3060 (12 GiB
   VRAM)
2. An HP Victus laptop PC with a GeForce RTX 3050 Ti (4 GiB VRAM)

### First, Install Ubuntu 22.04 WSL Instance

This is quite simple to do. At a command prompt: `wsl --install -d Ubuntu-22.04`

Ubuntu 22.04 is a preference of the author, not a requirement. Almost any instance
type with a modern enough Python version will work just fine. However, the
NVIDIA GPU section below has only been tested on Ubuntu 22.04 instances.

Once the environment started for the first time, WSL works just like a native
Ubuntu installation. If you have an NVIDIA device supporting CUDA, you can even
use it to run models faster.

###Setting Up CUDA for Onprem on Your WSL Instance

1. Install NVIDIA drivers. Instructions are [here](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl), but I used [Nvidia GeForce Experience](https://www.nvidia.com/en-us/geforce/geforce-experience/) to accomplish this.
2. Check `nvidia-smi` output in Ubuntu. Verify that it shows a device with Nvidia
   driver supporting CUDA 12.x. You can skip the rest of the step and follow the
   [Docker instructions](#how-to-run-onprem-in-a-docker-container) if you intend to
   use this WSL2 environment as a Docker host for *onprem* containers.
3. `apt install build-essential`
4. Do all [runfile (local) installation](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local)
   of the Official CUDA 12.x toolkit install for WSL2. Ensure the following
   warnings from the final summary output get addressed (If not done, you may
   see a "No CUDA-capable device detected" when attempting to run onprem with
   GPU layers.):
   * PATH includes `/usr/local/cuda-12.3/bin`
   * `LD_LIBRARY_PATH` includes `/usr/local/cuda-12.3/lib64`, or, add
     `/usr/local/cuda-12.3/lib64` to `/etc/ld.so.conf` and run `ldconfig` as root
5. `apt install python3.10-dev python3.10-venv`
6. Make your venv using the python 3.10 executable (should be `python3` on your
   *PATH*). I used the `--system-site-packages` option, though this may not have
   been necessary.
7. Activate your new venv.
8. Install llama-cpp-python with cuBLAS support:
   `CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python`
9. `pip install onprem`
10. Run an onprem prompt with `n_gpu_layers=-1`. This may be too much for a 4
    GiB GPU. I found `n_gpu_layers=31` fits in memory, and didn't go any higher.

