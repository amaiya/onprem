import os, yaml
import numpy as np
from pathlib import Path
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from sentence_transformers import SentenceTransformer, util
from onprem import LLM, utils as U

DATADIR = U.get_datadir()
DEFAULT_PROMPT = "List three cute names for a cat."
DEFAULT_YAML = """
llm:
  # model url (or model file name if previously downloaded)
  model_url: https://huggingface.co/TheBloke/WizardLM-13B-V1.2-GGUF/resolve/main/wizardlm-13b-v1.2.Q4_K_M.gguf
  # number of layers offloaded to GPU
  n_gpu_layers: 32
  # path to vector db folder
  vectordb_path: {datadir}/vectordb
  # path to model download folder
  model_download_path: {datadir}
  # number of source documents used by LLM.ask and LLM.chat
  rag_num_source_docs: 6
  # minimum similarity score for source to be considered by LLM.ask/LLM.chat
  rag_score_threshold: 0.0
ui:
  # title of application
  title: OnPrem.LLM
  # subtitle in "Talk to Your Documents" screen
  rag_title:
  # path to folder containing raw documents (used to construct direct links to document sources)
  rag_source_path:
  # base url (used to construct direct links to document sources)
  rag_base_url:
"""
DEFAULT_YAML_FNAME = 'webapp.yml'
DEFAULT_YAML_FPATH = os.path.join(DATADIR, DEFAULT_YAML_FNAME)

def write_default_yaml():
    """
    write default webapp.yml
    """
    yaml_content = DEFAULT_YAML.format(datadir=U.get_datadir()).strip()
    with open(DEFAULT_YAML_FPATH, 'w') as f:
        f.write(yaml_content)
    return

def read_config():
    """
    Read config file.  Returns a dictionary of the configuration and a boolean indicating whether or not a new config was created.
    """
    exists = os.path.exists(DEFAULT_YAML_FPATH)
    if not exists:
        write_default_yaml()
    with open(DEFAULT_YAML_FPATH, 'r') as stream:
        cfg = yaml.safe_load(stream)
    return cfg, not exists

@st.cache_resource
def get_llm():
    llm_config = read_config()[0]['llm']
    return LLM(confirm=False, **llm_config)


@st.cache_resource
def get_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


def compute_similarity(sentence1, sentence2):
    model = get_embedding_model()
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2)
    return cosine_score.cpu().numpy()[0][0]


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
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
    llm = get_llm()
    chat_box = st.empty()
    stream_handler = StreamHandler(chat_box, display_method='write')
    _ = llm.load_llm()
    llm.llm.callbacks = [stream_handler]
    return llm

def construct_link(filepath, source_path=None, base_url=None):
    """
    constructs a link to a document
    """
    import urllib
    filename = os.path.basename(filepath)
    if source_path is None:
        return filename
    base_url = base_url or '/'
    relative = str(Path(filepath).relative_to(source_path))
    link = os.path.join(base_url, relative) 
    return f'<a href="{urllib.parse.quote(link)}" ' +\
           f'target="_blank" title="Click to view original source">{filename}</a>'

def main():
    # Page setup
    cfg, cfg_was_created = read_config()

    TITLE  = cfg.get('ui', {}).get('title', 'OnPrem.LLM')
    RAG_TITLE = cfg.get('ui', {}).get('rag_title', None)
    APPEND_TO_PROMPT = cfg.get('prompt', {}).get('append_to_prompt', '')
    RAG_TEXT = None
    if os.path.exists(os.path.join(U.get_datadir(), 'rag_text.md')):
        with open(os.path.join(U.get_datadir(), 'rag_text.md'), 'r') as f:
            RAG_TEXT = f.read()
    RAG_SOURCE_PATH = cfg.get('ui', {}).get('rag_source_path', None)
    RAG_BASE_URL = cfg.get('ui', {}).get('rag_base_url', None)

    st.set_page_config(page_title=TITLE, page_icon="🐍", layout="wide")
    st.title(TITLE)
    if cfg_was_created:
        st.warning(f'No {DEFAULT_YAML_FNAME} file was found in {DATADIR}, so a default one was created for you. Please edit as necessary.')

    screen = st.sidebar.radio("Choose a Screen:",
                              ("Talk to Your Documents", "Use Prompts to Solve Problems"))
    st.sidebar.markdown(f"**Curent Model:**")
    st.sidebar.markdown(f"*{os.path.basename(cfg['llm']['model_url'])}*")
    if screen == 'Talk to Your Documents':
        st.sidebar.markdown('**Note:** Be sure to check any displayed sources to guard against hallucinations in answers.')
        if RAG_TITLE:
            st.header(RAG_TITLE)
        if RAG_TEXT:
            st.markdown(RAG_TEXT, unsafe_allow_html=True)
        question = st.text_input("Enter a question and press the `Ask` button:", value="", 
                                 help="Tip: If you don't like the answer quality after pressing 'Ask', try pressing the Ask button a second time. "
                                      "You can also try re-phrasing the question.")
        ask_button = st.button("Ask")
        llm = setup_llm()

        if question and ask_button:
            question = question + ' '+ APPEND_TO_PROMPT
            print(question)
            answer, docs = llm.ask(question)
            unique_sources = set()
            for doc in docs:
                answer_score = compute_similarity(answer, doc.page_content)
                question_score = compute_similarity(question, doc.page_content)
                if answer_score < 0.5 or question_score < 0.3: continue
                unique_sources.add( (doc.metadata['source'],
                                     doc.metadata.get('page', None), doc.page_content, question_score, answer_score))
            unique_sources = list(unique_sources)
            unique_sources.sort(key=lambda tup: tup[-1], reverse=True)
            if unique_sources:
                st.markdown('**One or More of These Sources Were Used to Generate the Answer:**')
                st.markdown('*You should inspect these sources to guard against hallucinations in the answer.*')
                for source in unique_sources:
                    fname = source[0]
                    fname = construct_link(fname, source_path=RAG_SOURCE_PATH, base_url = RAG_BASE_URL)
                    page = source[1] +1 if isinstance(source[1], int) else source[1]
                    content = source[2]
                    question_score = source[3]
                    answer_score = source[4]
                    st.markdown(f"- {fname} {', page '+str(page) if page else ''} : score: {answer_score}", help=f'{content}... [QUESTION_TO_SOURCE_SIMILARITY: {question_score})', 
                                unsafe_allow_html=True)
            elif "I don't know" not in answer:
                st.warning('No sources met the criteria to be displayed. This suggests the model may not be generating answers directly from your documents '+\
                           'and increases the likelihood of false information in the answer. ' +\
                           'You should be more cautious when using this answer.')
    else:
        prompt = st.text_area('Submit a Prompt to the LLM:', '', height=100, placeholder=DEFAULT_PROMPT)
        submit_button = st.button("Submit")
        st.markdown('*Examples of using prompts to solve different problems are [here](https://amaiya.github.io/onprem/examples.html).*')
        st.markdown('---')
        llm = setup_llm()
        if prompt and submit_button:
            saved_output = llm.prompt(prompt)


if __name__ == "__main__":
    main()
