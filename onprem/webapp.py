import os, yaml
import numpy as np
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
  model_url: https://huggingface.co/TheBloke/WizardLM-13B-V1.2-GGML/resolve/main/wizardlm-13b-v1.2.ggmlv3.q4_0.bin
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


def main():
    # Page setup
    cfg, cfg_was_created = read_config()

    TITLE  = cfg.get('streamlit', {}).get('title', 'OnPrem.LLM')
    RAG_TITLE = cfg.get('streamlit', {}).get('rag_title', None)
    APPEND_TO_PROMPT = cfg.get('prompt', {}).get('append_to_prompt', '')

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
            st.header(cfg['streamlit']['rag_title'])
        question = st.text_input("Enter a question and press the `Ask` button:", value="")
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
                unique_sources.add( (os.path.basename(doc.metadata['source']),
                                     doc.metadata.get('page', None), doc.page_content, question_score, answer_score))
            unique_sources = list(unique_sources)
            unique_sources.sort(key=lambda tup: tup[-1], reverse=True)
            if unique_sources:
                st.markdown('**One or More of These Sources Were Used to Generate the Answer:**')
                st.markdown('*You should inspect these sources to guard against hallucinations in the answer.*')
                for source in unique_sources:
                    fname = source[0]
                    page = source[1] +1 if isinstance(source[1], int) else source[1]
                    content = source[2]
                    question_score = source[3]
                    answer_score = source[4]
                    st.markdown(f"- {fname} {', page '+str(page) if page else ''} : score: {answer_score}", help=f'{content}... [QUESTION_TO_SOURCE_SIMILARITY: {question_score})')
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