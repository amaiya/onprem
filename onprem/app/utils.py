import os
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from langchain.callbacks.base import BaseCallbackHandler
from onprem import LLM, utils as U


def hide_webapp_sidebar_item():
    """
    Hides the webapp.py item from the sidebar navigation
    """
    hide_webapp_style = """
        <style>
            [data-testid="stSidebarNav"] ul li:first-child {
                display: none;
            }
        </style>
    """
    st.markdown(hide_webapp_style, unsafe_allow_html=True)


@st.cache_resource
def load_llm():
    """
    Load the LLM model with caching
    """
    from onprem.app.webapp import read_config
    llm_config = read_config()[0]["llm"]
    return LLM(confirm=False, **llm_config)


@st.cache_resource
def get_embedding_model():
    """
    Load the embedding model with caching
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def compute_similarity(sentence1, sentence2):
    """
    Compute cosine similarity between two sentences
    """
    model = get_embedding_model()
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2)
    return cosine_score.cpu().numpy()[0][0]


class StreamHandler(BaseCallbackHandler):
    """
    Callback handler for streaming LLM responses
    """
    def __init__(self, container, initial_text="", display_method="markdown"):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token + ""
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")


def setup_llm():
    """
    Set up the LLM with a stream handler
    """
    llm = load_llm()
    chat_box = st.empty()
    stream_handler = StreamHandler(chat_box, display_method="write")
    _ = llm.load_llm()
    llm.llm.callbacks = [stream_handler]
    return llm


def check_create_symlink(source_path, base_url):
    """
    Symlink to folder named <n> in datadir from streamlit's static folder
    """
    if base_url or not source_path:
        return source_path, base_url

    # set new source path
    new_source_path = os.path.dirname(source_path)
    symlink_name = os.path.basename(source_path)

    # check for existence
    staticdir = os.path.join(os.path.dirname(st.__file__), "static")
    if os.path.islink(os.path.join(staticdir, symlink_name)):
        return new_source_path, symlink_name

    # attempt creation
    try:
        os.symlink(new_source_path, os.path.join(staticdir, symlink_name))
    except Exception:
        return source_path, base_url
    return new_source_path, base_url


def construct_link(filepath, source_path=None, base_url=None):
    """
    Constructs a link to a document
    """
    import urllib
    from pathlib import Path

    filename = os.path.basename(filepath)
    if source_path is None:
        return filename
    base_url = base_url or "/"
    relative = str(Path(filepath).relative_to(source_path))
    link = os.path.join(base_url, relative)
    return (
        f'<a href="{urllib.parse.quote(link)}" '
        + f'target="_blank" title="Click to view original source">{filename}</a>'
    )