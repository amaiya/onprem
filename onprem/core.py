"""Core functionality for `onprem`"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_core.ipynb.

# %% auto 0
__all__ = ['MIN_MODEL_SIZE', 'MISTRAL_MODEL_URL', 'MISTRAL_MODEL_ID', 'MISTRAL_PROMPT_TEMPLATE', 'ZEPHYR_MODEL_URL',
           'ZEPHYR_MODEL_ID', 'ZEPHYR_PROMPT_TEMPLATE', 'LLAMA_MODEL_URL', 'LLAMA_MODEL_ID', 'LLAMA_PROMPT_TEMPLATE',
           'MODEL_URL_DICT', 'MODEL_ID_DICT', 'LLAMA_CPP', 'TRANSFORMERS', 'ENGINE_DICT', 'PROMPT_DICT',
           'DEFAULT_MODEL', 'DEFAULT_ENGINE', 'DEFAULT_EMBEDDING_MODEL', 'DEFAULT_QA_PROMPT',
           'AnswerConversationBufferMemory', 'LLM']

# %% ../nbs/00_core.ipynb 3
from . import utils as U
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI, AzureChatOpenAI
import os
import warnings
from typing import Any, Dict, Optional, Callable

# %% ../nbs/00_core.ipynb 4
# reference: https://github.com/langchain-ai/langchain/issues/5630#issuecomment-1574222564
class AnswerConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        return super(AnswerConversationBufferMemory, self).save_context(
            inputs, {"response": outputs["answer"]}
        )

MIN_MODEL_SIZE = 250000000
MISTRAL_MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MISTRAL_MODEL_ID = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
MISTRAL_PROMPT_TEMPLATE = "[INST] {prompt} [/INST]"
ZEPHYR_MODEL_URL = "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf"
ZEPHYR_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
ZEPHYR_PROMPT_TEMPLATE = "<|system|>\n</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"
LLAMA_MODEL_URL = "https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
LLAMA_MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct"
LLAMA_PROMPT_TEMPLATE = """<|start_header_id|>system<|end_header_id|>

You are a super-intelligent helpful assistant that executes instructions.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
MODEL_URL_DICT = {'mistral' : MISTRAL_MODEL_URL, 'zephyr':ZEPHYR_MODEL_URL, 'llama' : LLAMA_MODEL_URL}
MODEL_ID_DICT = {'mistral': MISTRAL_MODEL_ID, 'zephyr' : ZEPHYR_MODEL_ID, 'llama': LLAMA_MODEL_ID}
LLAMA_CPP = 'llama.cpp'
TRANSFORMERS = 'transformers'
ENGINE_DICT = {LLAMA_CPP: MODEL_URL_DICT, TRANSFORMERS : MODEL_ID_DICT}
PROMPT_DICT = {'mistral': MISTRAL_PROMPT_TEMPLATE, 'zephyr' : ZEPHYR_PROMPT_TEMPLATE, 'llama' : LLAMA_PROMPT_TEMPLATE} 
DEFAULT_MODEL = 'mistral'
DEFAULT_ENGINE = LLAMA_CPP
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_QA_PROMPT = """"Use the following pieces of context delimited by three backticks to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

```{context}```

Question: {question}
Helpful Answer:"""


class LLM:
    def __init__(
        self,
        model_url:Optional[str] = None,
        model_id:Optional[str] = None,
        default_model:str  = DEFAULT_MODEL,
        default_engine:str = DEFAULT_ENGINE,
        n_gpu_layers: Optional[int] = None,
        prompt_template: Optional[str] = None,
        model_download_path: Optional[str] = None,
        vectordb_path: Optional[str] = None,
        max_tokens: int = 512,
        n_ctx: int = 3900,
        n_batch: int = 1024,
        stop:list=[],
        mute_stream: bool = False,
        callbacks=[],
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_model_kwargs: dict = {"device": "cpu"},
        embedding_encode_kwargs: dict = {"normalize_embeddings": False},
        rag_num_source_docs: int = 4,
        rag_score_threshold: float = 0.0,
        check_model_download:bool=True,
        confirm: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        """
        LLM Constructor.  Extra `kwargs` (e.g., temperature) are fed directly to `langchain.llms.LlamaCpp` 
        or `langchain.hugging_face.HuggingFacePipeline`.

        **Args:**

        - *model_url*: URL to `.GGUF` model (or the filename if already been downloaded to `model_download_path`).
                       To use an OpenAI-compatible REST API (e.g., vLLM, OpenLLM, Ollama), supply the URL (e.g., `http://localhost:8080/v1`).
                       To use a cloud-based OpenAI model, replace URL with: `openai://<name_of_model>` (e.g., `openai://gpt-3.5-turbo`).
                       To use Azure OpenAI, replace URL with: with: `azure://<deployment_name>`.
                       If None, use the model indicated by `default_model`.
        - *model_id*: Name of or path to Hugging Face model (e.g., in SafeTensor format). Hugging Face Transformers is used for LLM generation instead of **llama-cpp-python**. Mutually-exclusive with `model_url` and `default_model`. The `n_gpu_layers` and `model_download_path` parameters are ignored if `model_id` is supplied.
        - *default_model*: One of {'mistral', 'zephyr', 'llama'}, where mistral is Mistral-Instruct-7B-v0.2, zephyr is Zephyr-7B-beta, and llama is Llama-3.1-8B.
        - *default_engine*: The engine used to run the `default_model`. One of {'llama.cpp', 'transformers'}.
        - *n_gpu_layers*: Number of layers to be loaded into gpu memory. Default is `None`.
        - *prompt_template*: Optional prompt template (must have a variable named "prompt"). Prompt templates are not typically needed when using the `model_id` parameter, as transformers sets it automatically.
        - *model_download_path*: Path to download model. Default is `onprem_data` in user's home directory.
        - *vectordb_path*: Path to vector database (created if it doesn't exist).
                           Default is `onprem_data/vectordb` in user's home directory.
        - *max_tokens*: The maximum number of tokens to generate.
        - *n_ctx*: Token context window. (Llama2 models have max of 4096.)
        - *n_batch*: Number of tokens to process in parallel.
        - *stop*: a list of strings to stop generation when encountered (applied to all calls to `LLM.prompt`)
        - *mute_stream*: Mute ChatGPT-like token stream output during generation
        - *callbacks*: Callbacks to supply model
        - *embedding_model_name*: name of sentence-transformers model. Used for `LLM.ingest` and `LLM.ask`.
        - *embedding_model_kwargs*: arguments to embedding model (e.g., `{device':'cpu'}`).
        - *embedding_encode_kwargs*: arguments to encode method of
                                     embedding model (e.g., `{'normalize_embeddings': False}`).
        - *rag_num_source_docs*: The maximum number of documents retrieved and fed to `LLM.ask` and `LLM.chat` to generate answers
        - *rag_score_threshold*: Minimum similarity score for source to be considered by `LLM.ask` and `LLM.chat`
        - *confirm*: whether or not to confirm with user before downloading a model
        - *verbose*: Verbosity
        """
        self.model_id = None
        self.model_url = None
        if model_url and model_id:
            raise ValueError('The parameters model_url and model_id are mutually-exclusive.')
        elif model_id:
            self.model_id = model_id
            self.model_name = os.path.basename(model_id)
        elif model_url:
            self.model_url = model_url.split("?")[0]
            self.model_name = os.path.basename(model_url)
            self.model_url = MODEL_DICT[default_model] if not model_url else model_url
        else: # neither supplied so use defaults
            url_or_id = ENGINE_DICT[default_engine][default_model]
            self.model_name = os.path.basename(url_or_id)
            if default_engine == LLAMA_CPP:
                self.model_url = url_or_id
                self.model_id = None
                prompt_template = PROMPT_DICT[default_model] if not prompt_template else prompt_template
            else:
                self.model_url = None
                self.model_id = url_or_id
        self.model_download_path = model_download_path or U.get_datadir()

        if self.is_llamacpp():
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ValueError('To run local LLMs, the llama-cpp-python package is required. ' +\
                                 'You can visit https://python.langchain.com/docs/integrations/llms/llamacpp ' +\
                                 'and follow the instructions for your operating system.')
        if self.is_llamacpp() and not os.path.isfile(os.path.join(self.model_download_path, self.model_name)):
            self.download_model(
                self.model_url,
                model_download_path=self.model_download_path,
                confirm=confirm,
            )
        self.prompt_template = prompt_template
        self.vectordb_path = vectordb_path
        self.llm = None
        self.ingester = None
        self.qa = None
        self.chatqa = None
        self.n_gpu_layers = n_gpu_layers
        self.max_tokens = max_tokens
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.stop = stop
        self.mute_stream = mute_stream
        self.callbacks = [] if mute_stream else [StreamingStdOutCallbackHandler()]
        if callbacks:
            self.callbacks.extend(callbacks)
        self.embedding_model_name = embedding_model_name
        self.embedding_model_kwargs = embedding_model_kwargs
        self.embedding_encode_kwargs = embedding_encode_kwargs
        self.rag_num_source_docs = rag_num_source_docs
        self.rag_score_threshold = rag_score_threshold
        self.check_model_download = check_model_download
        self.verbose = verbose
        self.extra_kwargs = kwargs


        # explicitly set offload_kqv
        # reference: https://github.com/abetlen/llama-cpp-python/issues/999#issuecomment-1858041458
        # commented out now: no longer needed as of llama-cpp-python==0.2.75
        #self.offload_kqv = True if n_gpu_layers is not None and n_gpu_layers > 0 else False

        # load LLM
        self.load_llm()

        # issue warning
        if self.is_openai_model():
            warnings.warn(f'The model you supplied is {self.model_name}, an external service (i.e., not on-premises). '+\
                          'Use with caution, as your data and prompts will be sent externally.')

    def is_openai_model(self):
        return self.model_url and self.model_url.lower().startswith('openai')

    def is_azure(self):
        return self.model_url and self.model_url.lower().startswith('azure')


    def is_local_api(self):
        basename = os.path.basename(self.model_url) if self.model_url else None
        return self.model_url and self.model_url.lower().startswith('http') and not basename.lower().endswith('.gguf') and not basename.lower().endswith('.bin')

    def is_local(self):
        return not self.is_openai_model() and not self.is_local_api()

    def is_llamacpp(self):
        return self.is_local() and not self.is_hf()

    def is_hf(self):
        return self.model_id is not None

    def update_max_tokens(self, value:int=512):
        """
        Update `max_tokens` (maximum length of generation).
        """
        llm = self.load_llm()
        llm.max_tokens = value


    def update_stop(self, value:list=[]):
        """
        Update `max_tokens` (maximum length of generation).
        """
        llm = self.load_llm()
        llm.stop = value


    @classmethod
    def download_model(
        cls,
        model_url:Optional[str] = None,
        default_model:str=DEFAULT_MODEL,
        model_download_path: Optional[str] = None,
        confirm: bool = True,
        ssl_verify: bool = True,
    ):
        """
        Download an LLM in GGML format supported by [lLama.cpp](https://github.com/ggerganov/llama.cpp).

        **Args:**

        - *model_url*: URL of model. If None, then use default_model.
        - *default_model*: One of {'mistral', 'zephyr', 'llama'}, where mistral is Mistral-Instruct-7B-v0.2, zephyr is Zephyr-7B-beta, and llama is Llama-3.1-8B.
        - *model_download_path*: Path to download model. Default is `onprem_data` in user's home directory.
        - *confirm*: whether or not to confirm with user before downloading
        - *ssl_verify*: If True, SSL certificates are verified.
                        You can set to False if corporate firewall gives you problems.
        """
        model_url = MODEL_URL_DICT[default_model] if not model_url else model_url
        if 'https://huggingface.co' in model_url and 'resolve' not in model_url:
            warnings.warn('\n\nThe supplied URL may not be pointing to the actual GGUF model file.  Please check it.\n\n')
        datadir = model_download_path or U.get_datadir()
        model_name = os.path.basename(model_url)
        filename = os.path.join(datadir, model_name)
        confirm_msg = f"\nYou are about to download the LLM {model_name} to the {datadir} folder. Are you sure?"
        if os.path.isfile(filename):
            confirm_msg = f"There is already a file {model_name} in {datadir}.\n Do you want to still download it?"

        shall = True
        if confirm:
            shall = input("%s (y/N) " % confirm_msg).lower() == "y"
        if shall:
            U.download(model_url, filename, verify=ssl_verify)
        else:
            warnings.warn(
                f'{model_name} was not downloaded because "Y" was not selected.'
            )
        return

    def load_ingester(self):
        """
        Get `Ingester` instance.
        You can access the `langchain_chroma.Chroma` instance with `load_ingester().get_db()`.
        """
        if not self.ingester:
            from onprem.ingest import Ingester

            self.ingester = Ingester(
                embedding_model_name=self.embedding_model_name,
                embedding_model_kwargs=self.embedding_model_kwargs,
                embedding_encode_kwargs=self.embedding_encode_kwargs,
                persist_directory=self.vectordb_path,
            )
        return self.ingester

    def load_vectordb(self):
        """
        Get Chroma db instance
        """
        ingester = self.load_ingester()
        db = ingester.get_db()
        if not db:
            raise ValueError(
                "A vector database has not yet been created. Please call the LLM.ingest method."
            )
        return db

    def ingest(
        self,
        source_directory: str, # path to folder containing documents
        chunk_size: int = 500, # text is split to this many characters by `langchain.text_splitter.RecursiveCharacterTextSplitter`
        chunk_overlap: int = 50, # character overlap between chunks in `langchain.text_splitter.RecursiveCharacterTextSplitter`
        ignore_fn:Optional[Callable] = None, # callable that accepts the file path and returns True for ignored files
        pdf_use_unstructured:bool=False, # If True, use unstructured for PDF extraction
        **kwargs, # Extra kwargs fed to `langchain_community.document_loaders.pdf.UnstructuredPDFLoader` when pdf_use_unstructured is True
    ):
        """
        Ingests all documents in `source_folder` into vector database.
        Previously-ingested documents are ignored.
        Extra kwargs fed directly to `langchain_community.document_loaders.pdf.UnstructuredPDFLoader` when pdf_use_unstructured is True.
        """
        ingester = self.load_ingester()
        return ingester.ingest(
            source_directory,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, ignore_fn=ignore_fn,
            pdf_use_unstructured=pdf_use_unstructured,
            **kwargs
        )

    def check_model(self):
        """
        Returns the path to the model
        """
        if not self.is_llamacpp():
            return None
        datadir = self.model_download_path
        model_path = os.path.join(datadir, self.model_name)
        if not os.path.isfile(model_path):
            raise ValueError(
                f"The LLM model {self.model_name} does not appear to have been downloaded. "
                + "Execute the download_model() method to download it."
            )
        return model_path

    def load_llm(self):
        """
        Loads the LLM from the model path.
        """

        if not self.llm and self.is_openai_model():
            self.llm = ChatOpenAI(model_name=self.model_name, 
                                  callbacks=self.callbacks, 
                                  streaming=not self.mute_stream,
                                  max_tokens=self.max_tokens,
                                  **self.extra_kwargs)
        elif not self.llm and self.is_azure():
            self.llm = AzureChatOpenAI(azure_deployment=self.model_name, 
                                  callbacks=self.callbacks, 
                                  streaming=not self.mute_stream,
                                  max_tokens=self.max_tokens,
                                  **self.extra_kwargs)

        elif not self.llm and self.is_local_api():
            self.llm = ChatOpenAI(base_url=self.model_url,
                                  #model_name=self.model_name, 
                                  callbacks=self.callbacks, 
                                  streaming=not self.mute_stream,
                                  max_tokens=self.max_tokens,
                                  **self.extra_kwargs)
        elif not self.llm and self.is_hf():
            # Hugging Face model
            from transformers import BitsAndBytesConfig, TextStreamer, AutoTokenizer
            from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
            import torch
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            streamer = TextStreamer(tokenizer)
    
            hfpipe = HuggingFacePipeline.from_model_id(
                model_id=self.model_id,
                task="text-generation",
                pipeline_kwargs=dict(
                     max_new_tokens=self.max_tokens,
                     #max_length = self.n_ctx, # cannot supply both
                     do_sample=True if self.extra_kwargs.get('temperature', 0.8)>0.0 else False ,
                     repetition_penalty=1.03,
                     return_full_text=False,
                    streamer=streamer,
                    **self.extra_kwargs,
                     #max_memory={0: "5GiB", "cpu": "30GiB"},
                ),
                model_kwargs={"quantization_config": quantization_config,
                              "torch_dtype": torch.bfloat16}
                )
            self.llm = ChatHuggingFace(llm=hfpipe)

        elif not self.llm:
            model_path = self.check_model()
            if self.check_model_download and os.path.getsize(model_path) < MIN_MODEL_SIZE:
                raise ValueError(f'The model file ({model_path} is less than {MIN_MODEL_SIZE} bytes. ' +\
                                 'It may not have been fully downloaded. Please delete the file and start again. ')
            self.llm = LlamaCpp(
                model_path=model_path,
                max_tokens=self.max_tokens,
                n_batch=self.n_batch,
                callbacks=self.callbacks,
                verbose=self.verbose,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                #offload_kqv = self.offload_kqv,
                **self.extra_kwargs,
            )

        return self.llm


    def prompt(self, prompt, prompt_template: Optional[str] = None, stop:list=[], **kwargs):
        """
        Send prompt to LLM to generate a response.
        Extra keyword arguments are sent directly to the model invocation.

        **Args:**

        - *prompt*: The prompt to supply to the model
        - *prompt_template*: Optional prompt template (must have a variable named "prompt").
                             This value will override any `prompt_template` value supplied 
                             to `LLM` constructor.
        - *stop*: a list of strings to stop generation when encountered. 
                  This value will override the `stop` parameter supplied to `LLM` constructor.

        """
        llm = self.load_llm()
        prompt_template = self.prompt_template if prompt_template is None else prompt_template
        if prompt_template:
            prompt = prompt_template.format(**{"prompt": prompt})
        stop = stop if stop else self.stop
        res = llm.invoke(prompt, stop=stop, **kwargs)
        return res.content if self.is_openai_model() or self.is_hf() else res



    def load_qa(self, prompt_template: str = DEFAULT_QA_PROMPT):
        """
        Prepares and loads the `langchain.chains.RetrievalQA` object

        **Args:**

        - *prompt_template*: A string representing the prompt with variables "context" and "question"
        """
        if self.qa is None:
            db = self.load_vectordb()
            retriever = db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.rag_num_source_docs,
                    "score_threshold": self.rag_score_threshold,
                },
            )
            llm = self.load_llm()
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            self.qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT},
            )
        return self.qa

    def load_chatqa(self):
        """
        Prepares and loads a `langchain.chains.ConversationalRetrievalChain` instance
        """
        if self.chatqa is None:
            db = self.load_vectordb()
            retriever = db.as_retriever(
                search_type="similarity_score_threshold",  # see note in constructor
                search_kwargs={
                    "k": self.rag_num_source_docs,
                    "score_threshold": self.rag_score_threshold,
                },
            )
            llm = self.load_llm()
            memory = AnswerConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )
            self.chatqa = ConversationalRetrievalChain.from_llm(
                llm, retriever, memory=memory, return_source_documents=True
            )
        return self.chatqa

    def ask(self, question: str, qa_template=DEFAULT_QA_PROMPT, prompt_template=None, **kwargs):
        """
        Answer a question based on source documents fed to the `ingest` method.
        Extra keyword arguments are sent directly to the model invocation.

        **Args:**

        - *question*: a question you want to ask
        - *qa_template*: A string representing the prompt with variables "context" and "question"
        - *prompt_template*: the model-specific template in which everything (including QA template) should be wrapped.
                            Should have a single variable "{prompt}". Overrides the `prompt_template` parameter supplied to 
                            `LLM` constructor.

        **Returns:**

        - A dictionary with keys: `answer`, `source_documents`, `question`
        """
        prompt_template = self.prompt_template if prompt_template is None else prompt_template
        prompt_template = qa_template if prompt_template is None else prompt_template.format(**{'prompt': qa_template})
        qa = self.load_qa(prompt_template=prompt_template)
        res = qa.invoke(question, **kwargs)
        res["question"] = res["query"]
        del res["query"]
        res["answer"] = res["result"]
        del res["result"]
        return res

    def chat(self, question: str, **kwargs):
        """
        Chat with documents fed to the `ingest` method.
        Unlike `LLM.ask`, `LLM.chat` includes conversational memory.
        Extra keyword arguments are sent directly to the model invocation.

        **Args:**

        - *question*: a question you want to ask

        **Returns:**

        - A dictionary with keys: `answer`, `source_documents`, `question`, `chat_history`
        """
        chatqa = self.load_chatqa()
        res = chatqa.invoke(question, **kwargs)
        return res
