FROM python:3.11
RUN pip --no-input install --upgrade pip
RUN pip --no-input install onprem
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
RUN pip --no-input install streamlit
