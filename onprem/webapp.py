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
def read_config():
    cfg_file = os.path.join(DATADIR, 'webapp.yml')
    if not os.path.exists(cfg_file):
        raise ValueError(f'There is no webapp.yml file in {DATADIR}. ' +\
                          'Please create one. An example webapp.yml file ' +\
                          'can be downloaded from here: https://raw.githubusercontent.com/amaiya/onprem/master/nbs/webapp.yml'
                         )
    with open(cfg_file, 'r') as stream:
        cfg = yaml.safe_load(stream)
    return cfg

@st.cache_resource
def get_llm():
    llm_config = read_config()['llm']
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
    cfg = read_config()

    TITLE  = cfg.get('streamlit', {}).get('title', 'OnPrem.LLM')
    RAG_TITLE = cfg.get('streamlit', {}).get('rag_title', None)
    APPEND_TO_PROMPT = cfg.get('prompt', {}).get('append_to_prompt', '')
    NUM_SOURCE_DOCS = cfg.get('streamlit', {}).get('num_source_docs', 4)

    st.set_page_config(page_title=TITLE, page_icon="🐍", layout="wide")
    st.title(TITLE)

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
            answer, docs = llm.ask(question, num_source_docs=NUM_SOURCE_DOCS)
            unique_sources = set()
            for doc in docs:
                score = compute_similarity(answer, doc.page_content)
                if score < 0.5: continue
                unique_sources.add( (os.path.basename(doc.metadata['source']),
                                     doc.metadata.get('page', None), doc.page_content, score))
            unique_sources = list(unique_sources)
            unique_sources.sort(key=lambda tup: tup[-1], reverse=True)
            if unique_sources:
                st.markdown('**One or More of These Sources Were Used to Generate the Answer:**')
                for source in unique_sources:
                    fname = source[0]
                    page = source[1] +1 if isinstance(source[1], int) else source[1]
                    content = source[2]
                    score = source[3]
                    st.markdown(f"- {fname} {', page '+str(page) if page else ''} : score: {score}", help=content)
            else:
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
