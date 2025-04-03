import os
import yaml
import streamlit as st
from pathlib import Path
import mimetypes
from onprem import utils as U
from onprem.app.utils import hide_webapp_sidebar_item, hide_manage_page

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
    Main entry point for the Streamlit multipage application
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
    
    # Hide webapp from the sidebar using CSS
    hide_webapp_sidebar_item()
    
    # Hide Manage page based on configuration
    hide_manage_page()
    
    # Display current model in sidebar
    st.sidebar.markdown("**Current Model:**")
    st.sidebar.markdown(f"*{os.path.basename(cfg['llm']['model_url'])}*")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with [OnPrem](https://github.com/amaiya/onprem)")
    
    # Redirect to Home page
    import importlib.util
    
    try:
        # Calculate the path to the Home page
        home_page_path = os.path.join(os.path.dirname(__file__), "pages", "0_Home.py")
        
        # Import the module
        spec = importlib.util.spec_from_file_location("home_page", home_page_path)
        home_page = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(home_page)
        
        # Run the Home page's main function
        home_page.main()
    except Exception as e:
        # If there's an error, show a simple redirect button
        st.title(TITLE)
        st.info("Please use the sidebar to navigate or click below to go to the Home page.")
        if st.button("Go to Home"):
            st.switch_page("pages/0_Home.py")


if __name__ == "__main__":
    main()
