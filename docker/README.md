How to Run Onprem in a Docker Container
=======================================

1. Confirm that your host has an Nvidia device and driver installed, supporting
   CUDA. On a Linux host, the `nvidia-smi` command-line utility can verify this.
   On a Windows host, launch *NVIDIA Control Panel*, and click on
   *System Information* in the lower-left-hand side of its window.
2. Install Docker. On Ubuntu Linux 22.04 (or WSL 2 running it),
   `apt install docker.io` will accomplish this.
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
       -v ~/onprem_data:/root/onprem_data onprem_cuda
   >>> from onprem import LLM
   >>> llm = LLM(n_gpu_layers=35)
   … et cetera …
   ```

You may use additional `-v` options to specify mapping a host folder to a folder
on the container if, e.g., you have a script utilizing *onprem* that you want
to execute. The *onprem_cuda* image does not include any facility for running
Jupyter notebooks.

If you wish to run the Streamlit application from the container, this command
will launch it on a given port, mapping that port to be accessible from the
host:

```shell
$ sudo docker run --gpus=all --cap-add SYS_RESOURCE -e USE_MLOCK=0 \
    -v ~/onprem_data:/root/onprem_data -p 8000:8000 onprem_cuda \
    onprem --port 8000
```

If *~/onprem_data/webapp.yml* already exists on your host machine, referring to
host paths under the `llm` key, you may need to modify the `llm.vectordb_path`
and `lm_model_download_path` keys as follows:

```yaml
vectordb_path: /root/onprem_data/vectordb
model_download_path: /root/onprem_data
```

If you wish to use the corpus query feature, you will need to define an additional volume
mapping the folders defined by `ui.reg_text_path` and `ui.rag_source_path` to, e.g.,
*/root/text-docs* and */root/raw-docs* in the container.
