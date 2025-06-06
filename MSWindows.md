# How to Install and Run OnPrem.LLM on Microsoft Windows

When using OnPrem.LLM on Microsoft Windows (e.g., Windows 11), you can either use  Windows Subsystem for Linux (WSL2) or Windows directly. This guide provides instructions for both.

## Using the System Python in Windows 11

1. Download and install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and make sure **Desktop development with C++** workload is selected and installed. This is needed to build `chroma-hnswlib` (as of this writing a pre-built wheel only exists for Python 3.11 and below). It is also needed if you need to build `llama-cpp-python` instead of installing a prebuilt wheel (as we do below).
2. Install Python 3.12:  Open "cmd" as administrator and type python to trigger *Microsoft Store* installation in Windows 11.
3. Create virtual environment: `python -m venv .venv`
4. Activate virtual environment: `.venv\Scripts\activate`. You can optionally append `C:\Users\<username\.venv\Scripts` to `Path` environment variable, so that you only need to type `activate` to enter virtual environment in the future.
5. Install PyTorch:
   - For CPU: `pip install torch torchvision torchaudio`
   - For GPU (if you do happen to have a GPU and have installed an [up-to-date NVIDIA driver](https://www.nvidia.com/en-us/drivers/) ): `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
     - Run the following at Python prompt to verify things are working. (The PyTorch binaries ship with all CUDA runtime dependencies and you don't need to locally install a CUDA toolkit or cuDNN, as long as NVIDIA driver is installed.)

     ```python
     In [1]: import torch

     In [2]: torch.cuda.is_available()
     Out[2]: True

     In [3]: torch.cuda.get_device_name()
     Out[3]: 'NVIDIA RTX A1000 6GB Laptop GPU'
     ```

6. Install an LLM Engine of your choice:

   **llama-cpp-python**:

   ```shell
   pip install llama-cpp-python==0.3.2 \
     --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
   ```
   If you want the newest version of `llama-cpp-python`, running `pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
` will build and install the latest version of the package.
   
   If you want the LLM to generate faster answers using your GPU, then you'll
   need to install the CUDA Toolkit from
   [here](https://developer.nvidia.com/cuda-12-6-0-download-archive?target_os=Windows).
   If you have issues, see
   [this guide](https://medium.com/@piyushbatra1999/installing-llama-cpp-python-with-nvidia-gpu-acceleration-on-windows-a-short-guide-0dfac475002d)
   for tips. You will also need to re-install `llama-cpp-python` with CUDA
   support, as described
   [here](https://python.langchain.com/docs/integrations/llms/llamacpp/#installation-with-windows):

   ```shell
   set FORCE_CMAKE=1
   set CMAKE_ARGS=-DGGML_CUDA=ON
   pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
   ```
   
    **Ollama**:
    - [Download and install Ollama](https://ollama.com/)
    - At a command prompt, pull a model: `ollama pull llama3.2`

    
    **Hugging Face Transformers**
    - Install the autoawq package: `pip install autoawq`, which is needed to load and run models quantized using AWQ in **transformers**.
  
    **Cloud LLM providers like OpenAI, Anthropic, and Amazon Bedrock**
   - No extra packages are required for cloud LLMs, but you must register an API key with the provider.

  
8. Install OnPrem.LLM: `pip install onprem[chroma]`
9. [OPTIONAL] If you're behind a corporate firewall and  have SSL certificate
   issues, you can try adding `REQUESTS_CA_BUNDLE` and `SSL_CERT_FILE` as
   environment variables and point them to the location of the certificate file
   for your organization, so hugging face models can be downloaded, etc.
10. Try onprem at a Python prompt to make sure it works. Run the `python`
    command and type the following:
   
    ```python
    from onprem import LLM
    # For llama-cpp-python, load LLM like this:
    llm = LLM() # to use GPU instead of CPU, use n_gpu_layers parameter: LLM(n_gpu_layers=-1)

    # For Ollama, load LLM like this:
    llm = LLM('ollama/llama3.2')

    # For transformers, load LLM like this:
    llm = LLM(default_engine="transformers", device='cuda')

    # For cloud LLM providers such as OpenAI
    llm = LLM('openai/gpt-4o-mini')
    
    # Try out a prompt
    llm.prompt('List three cute names for a cat.')

    # On a multi-core 2.5 GHz laptop CPU (e.g., 13th Gen Intel(R) Core(TM)
    # i7-13800H 2.50 GHz), you should get speeds of around 12 tokens per second.
    # If enabling GPU support as described above, speeds are much faster.
    ```
 

11. Try the [Web GUI](https://amaiya.github.io/onprem/webapp.html):
    - Start the Web app:  `onprem --port 8000` at a command prompt and clicking on the hyperlink.
    - **Using Ollama with Web App**: If using **Ollama** as the LLM engine, after starting the Web app for the first time, go to **Manage -> Configuration** and edit the configuration by removing the default `model_url`, replace with the following, and then press the **Save Configuration** button:
      ```yaml
      llm:
        model_url: ollama/llama3.2
        api_key: na
      ```
    - If using a cloud LLM with Web app, set the `model_url` in the configuration file accordingly (e.g., `openai/gpt-4o-mini`).
    -  **Changing the `store_type`**: You can optionally change to `store_type` from `dense` to `sparse` for faster document ingestion and easier installation. (Using the default `store_type="dense"` requires installation of `chromadb` and `langchain_chroma`.)
    -  **Changing the `max_tokens`**: You can increase `max_tokens` to say, 1024 or 2048, for longer LLM answers. (Not needed for Ollama, as Ollama sets a  higher value by default.)
    - After restarting the Web app, you will be able to interact with the LLM for:
       - Interactive chatting and prompting like ChatGPT
       - Document question-answering
       - Document analysis
       - Document search.
    - See the [Web UI documentation](https://amaiya.github.io/onprem/webapp.html) for more information.
    
## Using WSL2 (with GPU Acceleration)

### Install Ubuntu WSL2 Instance

This is quite simple to do. Example: At a command prompt: `wsl --install -d Ubuntu-24.04`

Once the environment started for the first time, WSL works just like a native
Ubuntu installation. If you have an NVIDIA device supporting CUDA, you can even
use it to run models faster.

### Installing OnPrem.LLM on WSL2

1. Install up-to-date NVIDIA drivers. Instructions are [here](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl).
2. Run these commands:

```sh
# Might be needed from a fresh install
sudo apt update
sudo apt upgrade

sudo apt install gcc python3-dev

# Might be needed, per: https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2
sudo apt-key del 7fa2af80

# From https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/cuda-repo-wsl-ubuntu-12-5-local_12.5.1-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-5-local_12.5.1-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-5-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-5

# If needed
sudo apt install python3.12-venv
python3 -m venv ./ai_env
source ./ai_env/bin/activate

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install and build llama-cpp-python
CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=all-major" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade

# Install OnPrem.LLM
pip install onprem
```


Reference: [Getting Started With CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl)
