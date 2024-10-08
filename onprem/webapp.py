import os, yaml
import numpy as np
from pathlib import Path
import mimetypes
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from sentence_transformers import SentenceTransformer, util
from onprem import LLM, utils as U

DATADIR = U.get_datadir()
DEFAULT_PROMPT = "List three cute names for a cat."
DEFAULT_YAML = """
llm:
  # model url (or model file name if previously downloaded)
  # if changing, be sure to update the prompt_template variable below
  model_url: https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf
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
  # verbosity of Llama.cpp
  verbose: TRUE
  # additional parameters added in the "llm" YAML section will be fed directly to LlamaCpp (e.g., temperature)
  #temperature: 0.0
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
  rag_source_path:
  # base url (leave blank unless you're running your own separate web server to serve source documents)
  rag_base_url:
"""
DEFAULT_YAML_FNAME = "webapp.yml"
DEFAULT_YAML_FPATH = os.path.join(DATADIR, DEFAULT_YAML_FNAME)


def write_default_yaml():
    """
    write default webapp.yml
    """
    yaml_content = DEFAULT_YAML.format(datadir=U.get_datadir()).strip()
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


@st.cache_resource
def load_llm():
    llm_config = read_config()[0]["llm"]
    return LLM(confirm=False, **llm_config)


@st.cache_resource
def get_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def compute_similarity(sentence1, sentence2):
    model = get_embedding_model()
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2)
    return cosine_score.cpu().numpy()[0][0]


class StreamHandler(BaseCallbackHandler):
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
    llm = load_llm()
    chat_box = st.empty()
    stream_handler = StreamHandler(chat_box, display_method="write")
    _ = llm.load_llm()
    llm.llm.callbacks = [stream_handler]
    return llm


def check_create_symlink(source_path, base_url):
    """
    Symlink to folder named <name> in datadir from streamlit's static folder
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
    except Exception as e:
        return source_path, base_url
    return new_source_path, base_url


def construct_link(filepath, source_path=None, base_url=None):
    """
    constructs a link to a document
    """
    import urllib

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


def is_txt(fpath):
    try:
        result = mimetypes.guess_type(fpath)
        return result[0] and result[0].startswith("text/")
    except:
        return False


def main():
    # Page setup
    cfg, cfg_was_created = read_config()

    TITLE = cfg.get("ui", {}).get("title", "OnPrem.LLM")
    RAG_TITLE = cfg.get("ui", {}).get("rag_title", None)
    APPEND_TO_PROMPT = cfg.get("prompt", {}).get("append_to_prompt", "")
    RAG_TEXT = None
    RAG_TEXT_PATH = cfg.get("ui", {}).get("rag_text_path", None)
    PROMPT_TEMPLATE = cfg.get("prompt", {}).get("prompt_template", None)
    if RAG_TEXT_PATH and os.path.isfile(RAG_TEXT_PATH) and is_txt(RAG_TEXT_PATH):
        with open(RAG_TEXT_PATH, "r") as f:
            RAG_TEXT = f.read()
    RAG_SOURCE_PATH = cfg.get("ui", {}).get("rag_source_path", None)
    RAG_BASE_URL = cfg.get("ui", {}).get("rag_base_url", None)
    RAG_SOURCE_PATH, RAG_BASE_URL = check_create_symlink(RAG_SOURCE_PATH, RAG_BASE_URL)

    st.set_page_config(page_title=TITLE, page_icon="🐍", layout="wide")
    st.title(TITLE)
    if cfg_was_created:
        st.warning(
            f"No {DEFAULT_YAML_FNAME} file was found in {DATADIR}, so a default one was created for you. Please edit as necessary."
        )

    screen = st.sidebar.radio(
        "Choose a Screen:", ("Talk to Your Documents", "Use Prompts to Solve Problems")
    )
    st.sidebar.markdown(f"**Curent Model:**")
    st.sidebar.markdown(f"*{os.path.basename(cfg['llm']['model_url'])}*")
    if screen == "Talk to Your Documents":
        st.sidebar.markdown(
            "**Note:** Be sure to check any displayed sources to guard against hallucinations in answers."
        )
        if RAG_TITLE:
            st.header(RAG_TITLE)
        if RAG_TEXT:
            st.markdown(RAG_TEXT, unsafe_allow_html=True)
        question = st.text_input(
            "Enter a question and press the `Ask` button:",
            value="",
            help="Tip: If you don't like the answer quality after pressing 'Ask', try pressing the Ask button a second time. "
            "You can also try re-phrasing the question.",
        )
        ask_button = st.button("Ask")
        llm = setup_llm()

        if question and ask_button:
            question = question + " " + APPEND_TO_PROMPT
            print(f'question : {question}')
            print(f'prompt_template: {PROMPT_TEMPLATE}')
            result = llm.ask(question, prompt_template=PROMPT_TEMPLATE)
            answer = result["answer"]
            docs = result["source_documents"]
            unique_sources = set()
            for doc in docs:
                answer_score = compute_similarity(answer, doc.page_content)
                question_score = compute_similarity(question, doc.page_content)
                if answer_score < 0.5 or question_score < 0.3:
                    continue
                unique_sources.add(
                    (
                        doc.metadata["source"],
                        doc.metadata.get("page", None),
                        doc.page_content,
                        question_score,
                        answer_score,
                    )
                )
            unique_sources = list(unique_sources)
            unique_sources.sort(key=lambda tup: tup[-1], reverse=True)
            if unique_sources:
                st.markdown(
                    "**One or More of These Sources Were Used to Generate the Answer:**"
                )
                st.markdown(
                    "*You can inspect these sources for more information and to also guard against hallucinations in the answer.*"
                )
                for source in unique_sources:
                    fname = source[0]
                    fname = construct_link(
                        fname, source_path=RAG_SOURCE_PATH, base_url=RAG_BASE_URL
                    )
                    page = source[1] + 1 if isinstance(source[1], int) else source[1]
                    content = source[2]
                    question_score = source[3]
                    answer_score = source[4]
                    st.markdown(
                        f"- {fname} {', page '+str(page) if page else ''} : score: {answer_score:.3f}",
                        help=f"{content}... (QUESTION_TO_SOURCE_SIMILARITY: {question_score:.3f})",
                        unsafe_allow_html=True,
                    )
            elif "I don't know" not in answer:
                st.warning(
                    "No sources met the criteria to be displayed. This suggests the model may not be generating answers directly from your documents "
                    + "and increases the likelihood of false information in the answer. "
                    + "You should be more cautious when using this answer."
                )
    else:
        prompt = st.text_area(
            "Submit a Prompt to the LLM:",
            "",
            height=100,
            placeholder=DEFAULT_PROMPT,
            help="Tip: If you don't like the response quality after pressing 'Submit', try pressing the button a second time. "
            "You can also try re-phrasing the prompt.",
        )
        submit_button = st.button("Submit")
        st.markdown(
            "*Examples of using prompts to solve different problems are [here](https://amaiya.github.io/onprem/examples.html).*"
        )
        st.markdown("---")
        llm = setup_llm()
        if prompt and submit_button:
            print(prompt)
            saved_output = llm.prompt(prompt, prompt_template=PROMPT_TEMPLATE)


if __name__ == "__main__":
    main()
