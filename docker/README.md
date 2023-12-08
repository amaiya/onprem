How to Run Onprem in a Docker Container
=======================================

Ignore these steps if you don't have an NVIDIA GPU, and skip to the
[next section]().

1. (Optional) Confirm that your host has an Nvidia device and driver installed, supporting
   CUDA. On a Linux host, the `nvidia-smi` command-line utility can verify this.
   On a Windows host, launch *NVIDIA Control Panel*, and click on
   *System Information* in the lower-left-hand side of its window.
2. Install Docker. On Ubuntu Linux 22.04 (or WSL 2 running it),
   `apt install docker.io` will accomplish this.
3. (Optional) Install the NVIDIA Container Toolkit. Full instructions, including necessary
   configuration steps, are at
   [this link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
4. (Optional) Test that the NVIDIA Container Toolkit is passing through the NVIDIA Driver
   to properly run containers by following the post-installation instructions
   titled *Running a Sample Workload*.
5. Build the Docker image(s) by running `docker/docker_build.sh`

Run OnPrem in a Containerized Python REPL
-----------------------------------------

The following shell commands will launch the image's python interpreter in a
container, mapping the host's persistent *~/onprem_data/* folder to the correct
location inside the container.

You may use additional `-v` options to specify mapping a host folder to a folder
on the container if, e.g., you have a script utilizing *onprem* that you want
to execute. The *onprem_cuda* image does not include any facility for running
Jupyter notebooks.

### CUDA-Enabled REPL

The `11.7.1` container tag may be different in your case.

```shell
$ sudo docker run --rm -it --gpus=all --cap-add SYS_RESOURCE -e USE_MLOCK=0 \
    -v ~/onprem_data:/root/onprem_data onprem_cuda:11.7.1
>>> from onprem import LLM
>>> llm = LLM(n_gpu_layers=35)
… et cetera …
```

### CPU-only REPL

```shell
$ sudo docker run --rm -it -v ~/onprem_data:/root/onprem_data onprem:cpu
>>> from onprem import LLM
>>> llm = LLM()
… et cetera …
```

Run the Onprem Streamlit App from a Container
---------------------------------------------

If you wish to run the Streamlit application from the container, the commands in
the subsections below will launch it on a port 800, mapping that port to be
accessible as port 8000 on the host.

If *~/onprem_data/webapp.yml* already exists on your host machine, referring to
host paths under the `llm` key, you may need to modify the `llm.vectordb_path`
and `lm_model_download_path` keys as follows:

```yaml
llm:
  vectordb_path: /root/onprem_data/vectordb
  model_download_path: /root/onprem_data
```

### CUDA-Enabled Web App

```shell
$ sudo docker run --gpus=all --cap-add SYS_RESOURCE -e USE_MLOCK=0 \
    -v ~/onprem_data:/root/onprem_data -p 8000:8000 onprem_cuda:11.7.1 \
    onprem --port 8000
```

If you wish to use the document query feature, you will need to define an
additional volume mapping the folder defined by
`ui.rag_source_path`. If you have your text corpus on the host at *~/rag_texts*,
and the `ui.rag_source_path` in you *webapp.yml* set as follows:

```yaml
ui:
  rag_source_path: /root/rag_texts
```

First, you launch directly into the default Python REPL, with the additional
volume mapping, so that you can run commands to ingest your documents:

```shell
$ sudo docker run --rm -it --gpus=all --cap-add SYS_RESOURCE -e USE_MLOCK=0 \
    -v ~/onprem_data:/root/onprem_data -v ~/rag_texts:/root/rag_texts \
    onprem_cuda:11.7.1
…
>>> from onprem import LLM
>>> llm = LLM(use_larger=True, n_gpu_layers=35)
>>> llm.ingest("/root/rag_texts")
```

Press `CTRL-D` when done to exit the containerized Python REPL. Then, launch the
web app, with the additional volume mapping is as follows:

```shell
$ sudo docker run --gpus=all --cap-add SYS_RESOURCE -e USE_MLOCK=0 \
    -v ~/onprem_data:/root/onprem_data -v ~/rag_texts:/root/rag_texts \
    -p 8000:8000 onprem_cuda:11.7.1 onprem --port 8000
```

If you omit the *rag_texts* volume mapping, you will still be able to query the
ingested knowledge from your documents, but any displayed hyperlinks to your
documents will not work.

### CPU-Only Web App

This is very similar to running the CUDA image. The GPU and system resource
options are not needed for running the  CPU-only image.

```shell
$ sudo docker run -v ~/onprem_data:/root/onprem_data -p 8000:8000 onprem_cuda \
    onprem --port 8000
```

or with documents to query:

```shell
$ sudo docker run -v ~/onprem_data:/root/onprem_data \
    -v ~/rag_texts:/root/rag_texts -p 8000:8000 onprem_cuda onprem --port 8000
```