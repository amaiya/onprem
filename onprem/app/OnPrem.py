import os
import yaml
import streamlit as st
from pathlib import Path
import mimetypes
from onprem import utils as U
from onprem.app.utils import load_llm

# https://github.com/VikParuchuri/marker/issues/442#issuecomment-2636393925
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

DATADIR = U.get_datadir()
DEFAULT_YAML_FNAME = "config.yml"
DEFAULT_WEBAPP_DIR = U.get_webapp_dir()
DEFAULT_YAML_FPATH = os.path.join(DEFAULT_WEBAPP_DIR, DEFAULT_YAML_FNAME)
DEFAULT_PROMPT = "List three cute names for a cat."
DEFAULT_YAML = """
llm:
  # model url (or model file name if previously downloaded)
  # if changing, be sure to update the prompt_template variable below
  model_url: https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf
  # number of layers offloaded to GPU
  n_gpu_layers: -1
  # type of vector store
  # ("dual" means both Chroma semantic searches and conventional keyword searches are supported)
  store_type: dual
  # path to vector db folder
  vectordb_path: {webapp_dir}/vectordb
  # path to model download folder
  model_download_path: {models_dir}
  # number of source documents used by LLM.ask and LLM.chat
  rag_num_source_docs: 6
  # minimum similarity score for source to be considered by LLM.ask/LLM.chat
  rag_score_threshold: 0.0
  # verbosity of Llama.cpp
  verbose: TRUE
  # additional parameters added in the "llm" YAML section will be fed directly to LlamaCpp (e.g., temperature)
  #temperature: 0.0
  # max_tokens: 2048
prompt:
  # The default prompt_template is specifically for the Zephyr-7B model.
  # It will need to be changed if you change the model_url above.
  prompt_template: <|system|>\\n</s>\\n<|user|>\\nPROMPT_VARIABLE</s>\\n<|assistant|>
ui:
  # title of application
  title: OnPrem.LLM
  # subtitle in "Talk to Your Documents" screen
  rag_title:
  # path to markdown file with contents that will be inserted below rag_title
  rag_text_path:
  # path to folder containing raw documents (i.e., absolute path of folder you supplied to LLM.ingest)
  rag_source_path: {webapp_dir}/documents
  # base url (leave blank unless you're running your own separate web server to serve source documents)
  rag_base_url:
  # whether to show the Manage page in the sidebar (TRUE or FALSE)
  show_manage: TRUE
"""


def write_default_yaml():
    """
    write default webapp.yml
    """
    yaml_content = DEFAULT_YAML.format(
        webapp_dir=U.get_webapp_dir(),
        models_dir=U.get_models_dir()
    ).strip()
    yaml_content = yaml_content.replace('PROMPT_VARIABLE', '{prompt}')
    with open(DEFAULT_YAML_FPATH, "w") as f:
        f.write(yaml_content)
    return


def read_config():
    """
    Read config file.  Returns a dictionary of the configuration and a boolean indicating whether or not a new config was created.
    """
    exists = os.path.exists(DEFAULT_YAML_FPATH)
    if not exists:
        write_default_yaml()
    with open(DEFAULT_YAML_FPATH, "r") as stream:
        cfg = yaml.safe_load(stream)
    return cfg, not exists


def is_txt(fpath):
    try:
        result = mimetypes.guess_type(fpath)
        return result[0] and result[0].startswith("text/")
    except:
        return False


def main():
    """
    Main entry point for the Streamlit application
    """
    # Page setup
    cfg, cfg_was_created = read_config()

    TITLE = cfg.get("ui", {}).get("title", "OnPrem.LLM")
    st.set_page_config(
        page_title=TITLE, 
        page_icon="🐍", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # First run configuration warning
    if cfg_was_created:
        st.warning(
            f"No {DEFAULT_YAML_FNAME} file was found in {DATADIR}, so a default one was created for you. Please edit as necessary."
        )
    
    # Home page content
    st.header("Welcome to OnPrem.LLM")
    
    # Load LLM to get model name
    ## 2025-04-09: commenting this out, a streamlit is reloading the again model in other pages
    #llm = load_llm()
    #model_name = llm.model_name
    
    # Main content
    st.markdown("""
    This application allows you to interact with an LLM in multiple ways:
    
    1. **Use Prompts to Solve Problems**: Submit prompts directly to the LLM model
    2. **Talk to Your Documents**: Ask questions about your documents using RAG technology
    3. **Document Analysis**: Apply custom prompts to document chunks and export results
    4. **Search Documents**: Search through your indexed documents using keywords or semantic search
    5. **Manage**: Upload documents, manage folders, and configure the application settings
    
    Use the sidebar to navigate between these different features.
    """)
    
    ## see 2025-04-09 comment above
    # Model information
    #st.subheader("Current Model Information")
    #st.markdown(f"**Model**: {model_name}")
    
    # Quick links
    st.subheader("Quick Links")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💬 Use Prompts", use_container_width=True):
            st.switch_page("pages/1_Prompts.py")
    
    with col2:
        if st.button("📄 Talk to Documents", use_container_width=True):
            st.switch_page("pages/2_Document_QA.py")
    
    with col3:
        if st.button("📊 Document Analysis", use_container_width=True):
            st.switch_page("pages/3_Document_Analysis.py")
    
    with col4:
        if st.button("🔍 Search Documents", use_container_width=True):
            st.switch_page("pages/4_Document_Search.py")
    
    # Additional information
    st.markdown("---")
    st.markdown("Built with [OnPrem](https://github.com/amaiya/onprem)")


if __name__ == "__main__":
    main()
