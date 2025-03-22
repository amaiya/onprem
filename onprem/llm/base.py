"""Core functionality for LLMs"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/00_llm.base.ipynb.

# %% auto 0
__all__ = ['MIN_MODEL_SIZE', 'MISTRAL_MODEL_URL', 'MISTRAL_MODEL_ID', 'MISTRAL_PROMPT_TEMPLATE', 'ZEPHYR_MODEL_URL',
           'ZEPHYR_MODEL_ID', 'ZEPHYR_PROMPT_TEMPLATE', 'LLAMA_MODEL_URL', 'LLAMA_MODEL_ID', 'LLAMA_PROMPT_TEMPLATE',
           'MODEL_URL_DICT', 'MODEL_ID_DICT', 'LLAMA_CPP', 'TRANSFORMERS', 'ENGINE_DICT', 'PROMPT_DICT',
           'DEFAULT_MODEL', 'DEFAULT_ENGINE', 'DEFAULT_EMBEDDING_MODEL', 'DEFAULT_QA_PROMPT',
           'AnswerConversationBufferMemory', 'LLM']

# %% ../../nbs/00_llm.base.ipynb 3
from ..utils import get_datadir, download, format_string
from . import helpers
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages.ai import AIMessage
from langchain_core.documents import Document
import os
import warnings
from typing import Any, Dict, Optional, Callable, Union, List

# %% ../../nbs/00_llm.base.ipynb 4
# reference: https://github.com/langchain-ai/langchain/issues/5630#issuecomment-1574222564
class AnswerConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        return super(AnswerConversationBufferMemory, self).save_context(
            inputs, {"response": outputs["answer"]}
        )

MIN_MODEL_SIZE = 250000000
MISTRAL_MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MISTRAL_MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
MISTRAL_PROMPT_TEMPLATE = "[INST] {prompt} [/INST]"
ZEPHYR_MODEL_URL = "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf"
ZEPHYR_MODEL_ID = "TheBloke/zephyr-7B-beta-AWQ"
ZEPHYR_PROMPT_TEMPLATE = "<|system|>\n</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"
LLAMA_MODEL_URL = "https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
LLAMA_MODEL_ID = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
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
        store_type:str='dense',
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
        LLM Constructor.  Extra `kwargs` (e.g., temperature) are fed directly
        to `langchain.llms.LlamaCpp`  (if `model_url` is supplied) or
        `transformers.pipeline` (if `model_id` is supplied).

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
        - *store_type*: One of `dense` for conventional vector database or `sparse`, a vector database
                        that stores documents as sparse vectors (i.e., keyword search engine).  
                        (Documents stored in sparse vector databases are converted to dense vectors at inference time 
                        when used with `LLM.ask`.)
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
        - *rag_num_source_docs*: The maximum number of documents retrieved and fed to `LLM.ask` and `LLM.chat` to generate answers.
        - *rag_score_threshold*: Minimum similarity score for source to be considered by `LLM.ask` and `LLM.chat`.
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
        self.model_download_path = model_download_path or get_datadir()

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
        self.store_type = store_type
        self.llm = None
        self.vectorstore = None
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

    def is_sparse_store(self):
        return self.store_type == 'sparse'


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
        datadir = model_download_path or get_datadir()
        model_name = os.path.basename(model_url)
        filename = os.path.join(datadir, model_name)
        confirm_msg = f"\nYou are about to download the LLM {model_name} to the {datadir} folder. Are you sure?"
        if os.path.isfile(filename):
            confirm_msg = f"There is already a file {model_name} in {datadir}.\n Do you want to still download it?"

        shall = True
        if confirm:
            shall = input("%s (y/N) " % confirm_msg).lower() == "y"
        if shall:
            download(model_url, filename, verify=ssl_verify)
        else:
            warnings.warn(
                f'{model_name} was not downloaded because "Y" was not selected.'
            )
        return


    def load_vectorstore(self):
        """
        Get `VectorStore` instance.
        You can access the `langchain_chroma.Chroma` instance with `load_vectorstore().get_db()`.
        """
        from onprem.ingest.stores import DenseStore, SparseStore
        if not self.vectorstore:
            store_cls = SparseStore if self.is_sparse_store() else DenseStore

            # stor spare store in its own subfolder in `vectordb_path`
            store_path = os.path.join(self.vectordb_path, 'sparse') if self.is_sparse() else self.vectordb_path

            # create vector store
            self.vectorstore = store_cls(
                embedding_model_name=self.embedding_model_name,
                embedding_model_kwargs=self.embedding_model_kwargs,
                embedding_encode_kwargs=self.embedding_encode_kwargs,
                persist_directory=store_path,
            )
        return self.vectorstore


    def load_vectordb(self):
        """
        Get Chroma db instance
        """
        vectorstore= self.load_vectorstore()
        db = vectorstore.get_db()
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
        batch_size:int=1000, # batch size used when processing documents(e.g, creating embeddings).
        **kwargs, # Extra kwargs fed to downstream functions, `load_single_document` and/or `load_documents`
    ):
        """
        Ingests all documents in `source_folder` into vector database.
        Previously-ingested documents are ignored.
        Extra kwargs fed to `load_single_document` and/or `load_docments`.
        """
        vectorstore = self.load_vectorstore()
        return vectorstore.ingest(
            source_directory,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, ignore_fn=ignore_fn,
            llm=kwargs['llm'] if 'llm' in kwargs else self,
            batch_size=batch_size,
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
            from transformers import pipeline, TextStreamer, AutoTokenizer
            from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
            tokenizer = self.extra_kwargs['tokenizer'] if 'tokenizer' in self.extra_kwargs\
                        else AutoTokenizer.from_pretrained(self.model_id)
            if 'tokenizer' in self.extra_kwargs:
                del self.extra_kwargs['tokenizer']
            streamer = TextStreamer(tokenizer)


            pipe = pipeline('text-generation',
                              self.model_id,
                              tokenizer=tokenizer,
                              streamer=streamer,
                              max_new_tokens = self.max_tokens,
                              return_full_text=False,
                              do_sample=True if\
                                     self.extra_kwargs.get('temperature', 0.8)>0.0 else False ,
                              **self.extra_kwargs)
            hfpipe = HuggingFacePipeline(pipeline=pipe)
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


    def _format_image_prompt(self, prompt:str, image_path_or_url:str):
        """
        Correctly format image prompt
        """
        from langchain_core.messages import HumanMessage
        import base64
        if not image_path_or_url.startswith('http'):
            with open(image_path_or_url, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            image_path_or_url = f"data:image/jpeg;base64,{image_data}"

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_path_or_url},
                },
            ],
        )
        prompt = [message]
        return prompt


    def _format_pydantic_prompt(self, prompt, pydantic_model):
        """
        Correctly format prompt for Pydantic model
        """
        parser = PydanticOutputParser(pydantic_object=pydantic_model)
        prompt_obj = PromptTemplate(
                     template="Answer the user query.\n{format_instructions}\n{prompt}\n",
                     input_variables=["prompt"],
                     partial_variables={"format_instructions": parser.get_format_instructions()},)
        return (prompt_obj.invoke({'prompt': prompt}).text, parser)


    def pydantic_prompt(self, 
                        prompt:str,
                        pydantic_model=None,
                        attempt_fix:bool= False,
                        fix_llm=None,
                        stop:list=[],
                        **kwargs):
        """
        Accept a prompt as string and Pydantic model describing the desired output.
        Output will be a Pydantic object in the requested format.


        **Args:**

        - *prompt*: The prompt to supply to the model.
                    Either a string or OpenAI-style list of dictionaries
                    representing messages (e.g., "human", "system").
        - *pydantic_model*: A Pydanatic model (sublass of `pydantic.BaseModel` that describes the desired output format.
                             Example:
                             ```python
                             from pydantic import BaseModel, Field
                             class Joke(BaseModel):
                                 setup: str = Field(description="question to set up a joke")
                                 punchline: str = Field(description="answer to resolve the joke")
                             ```
                           Output will be a desired Pydantic object.
                           If `put_format=None`, then output is a string.
        - *attempt_fix*: Use an LLM call in attempt to correct malformed or incomplete outputs
        - *fix_llm*:  LLM to use for fixing (e.g., `langchain_openai.ChatOpenAI()`). If `None`, then existing `LLM.llm` used.
        - *stop*: a list of strings to stop generation when encountered. 
                  This value will override the `stop` parameter supplied to `LLM` constructor.
        """
        # setup up prompt for output parsing
        prompt, output_parser = self._format_pydantic_prompt(prompt, pydantic_model)

        # generate output
        output = self.prompt(prompt, stop=stop, **kwargs)

        # set parser
        fix_llm = fix_llm if fix_llm else self.llm
        parser = OutputFixingParser.from_llm(parser=output_parser, llm=fix_llm)\
                if attempt_fix else output_parser

        # parse output into Pydantic class
        try:
            return parser.parse(output)
        except:
            print()
            print()
            warnings.warn('LLM output was malformed or incomplete, so returning raw string output.')
            print()
            return output


    def prompt(self,
               prompt:Union[str, List[Dict]],
               output_parser:Optional[Any]=None,
               image_path_or_url:Optional[str] = None,
               prompt_template: Optional[str] = None, stop:list=[], **kwargs):
        """
        Send prompt to LLM to generate a response.
        Extra keyword arguments are sent directly to the model invocation.

        **Args:**

        - *prompt*: The prompt to supply to the model.
                    Either a string or OpenAI-style list of dictionaries
                    representing messages (e.g., "human", "system").
        - *image_path_or_url*: Path or URL to an image file
        - *prompt_template*: Optional prompt template (must have a variable named "prompt").
                             This value will override any `prompt_template` value supplied 
                             to `LLM` constructor.
        - *stop*: a list of strings to stop generation when encountered. 
                  This value will override the `stop` parameter supplied to `LLM` constructor.

        """
        # load llm
        llm = self.load_llm()

        # prompt is a list of dictionaries reprsenting messages
        if isinstance(prompt, list):
            try:
                res = llm.invoke(prompt, stop=stop, **kwargs)
            except Exception as e: # stop param fails with GPT-4o vision prompts
                res = llm.invoke(prompt, **kwargs)
        # prompt is string
        else:
            if image_path_or_url:
                prompt = self._format_image_prompt(prompt, image_path_or_url)
                res = llm.invoke(prompt, **kwargs) # including stop causes errors in gpt-4o
            else:
                prompt_template = self.prompt_template if prompt_template is None else prompt_template
                if prompt_template:
                    prompt = format_string(prompt_template, prompt=prompt)
                stop = stop if stop else self.stop
                if self.is_hf():
                    tokenizer = llm.llm.pipeline.tokenizer
                    # FIX for #113/#114
                    prompt = [{'role':'user', 'content':prompt}] if tokenizer.chat_template else prompt
                    # Call HF pipeline directly instead of `invoke`
                    # since LangChain is not passing along stop_strings
                    # parameter to pipeline
                    res = llm.llm.pipeline(prompt,
                                           stop_strings=stop if stop else None,
                                           tokenizer=tokenizer,
                                           **kwargs)[0]['generated_text']
                else:
                    res = llm.invoke(prompt, stop=stop, **kwargs)
        return res.content if isinstance(res, AIMessage) else res


    def load_chatqa(self):
        """
        Prepares and loads a `langchain.chains.ConversationalRetrievalChain` instance
        """
        if self.chatqa is None:
            db = self.load_vectordb()
            retriever = db.as_retriever(
                search_type="similarity_score_threshold",
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



    def query(self,
              query:str, # query string
              k:int = 4, # max number of results to return
              score_threshold:float=0.0, # minimum score for document to be considered as answer source
              filters:Optional[Dict[str, str]] = None, # filter sources by metadata values using Chroma metadata syntax (e.g., {'table':True})
              where_document:Optional[Dict[str, str]] = None, # filter sources by document content in Chroma syntax (e.g., {"$contains": "Canada"})
              **kwargs):
        """
        Perform a semantic search of the vector DB
        """
        store = self.load_vectorstore()
        results = store.query(query, 
                              filters=filters,
                              where_document=where_document,
                              k = k, **kwargs)
        if not results: return []
        docs, scores = zip(*results)
        for doc, score in zip(docs, scores):
            simscore = 1 - score
            doc.metadata["score"] = 1-score

        return [d for d in docs if d.metadata['score'] >= score_threshold]


    def _ask(self,
            question: str, # question as sting
            contexts:Optional[list]=None, # optional lists of contexts to answer question. If None, retrieve from vectordb.
            qa_template=DEFAULT_QA_PROMPT, # question-answering prompt template to tuse
            filters:Optional[Dict[str, str]] = None, # filter sources by metadata values using Chroma metadata syntax (e.g., {'table':True})
            where_document:Optional[Dict[str, str]] = None, # filter sources by document content in Chroma syntax (e.g., {"$contains": "Canada"})
            k:Optional[int]=None, # Number of sources to consider.  If None, use `LLM.rag_num_source_docs`.
            score_threshold:Optional[float]=None, # minimum similarity score of source. If None, use `LLM.rag_score_threshold`.
            table_k:int=1, # maximum number of tables to consider when generating answer
            table_score_threshold:float=0.35, # minimum similarity score for table to be considered in answer
             **kwargs):
        """
        Answer a question based on source documents fed to the `LLM.ingest` method.
        Extra keyword arguments are sent directly to `LLM.prompt`.
        Returns a dictionary with keys: `answer`, `source_documents`, `question`
        """

        if not contexts:
            # query the vector db
            docs = self.query(question, filters=filters, where_document=where_document,
                              k=k if k else self.rag_num_source_docs,
                              score_threshold=score_threshold if score_threshold else self.rag_score_threshold)
            if table_k>0:
                table_filters = filters.copy() if filters else {}
                table_filters = dict(table_filters, table=True)
                table_docs = self.query(f'{question} (table)', 
                                        filters=table_filters, 
                                        where_document=where_document,
                                        k=table_k,
                                        score_threshold=table_score_threshold)
                if table_docs:
                    docs.extend(table_docs[:k])
            context = '\n\n'.join([d.page_content for d in docs])
        else:
            docs = [Document(page_content=c, metadata={'source':'<SUBANSWER>'}) for c in contexts]
            context = "\n\n".join(contexts)

        # setup prompt
        prompt = format_string(qa_template,
                                 question=question,
                                 context = context)

        # prompt LLM
        answer = self.prompt(prompt,**kwargs)

        # return answer
        res = {}
        res['question'] = question
        res['answer'] = answer
        res['source_documents'] = docs
        return res


    def ask(self,
            question: str, # question as sting
            selfask:bool=False, # If True, use an agentic Self-Ask prompting strategy.
            qa_template=DEFAULT_QA_PROMPT, # question-answering prompt template to tuse
            filters:Optional[Dict[str, str]] = None, # filter sources by metadata values using Chroma metadata syntax (e.g., {'table':True})
            where_document:Optional[Dict[str, str]] = None, # filter sources by document content in Chroma syntax (e.g., {"$contains": "Canada"})
            k:Optional[int]=None, # Number of sources to consider.  If None, use `LLM.rag_num_source_docs`.
            score_threshold:Optional[float]=None, # minimum similarity score of source. If None, use `LLM.rag_score_threshold`.
            table_k:int=1, # maximum number of tables to consider when generating answer
            table_score_threshold:float=0.35, # minimum similarity score for table to be considered in answer
             **kwargs):
        """
        Answer a question based on source documents fed to the `LLM.ingest` method.
        Extra keyword arguments are sent directly to `LLM.prompt`.
        Returns a dictionary with keys: `answer`, `source_documents`, `question`
        """

        if selfask and helpers.needs_followup(question, self):
            subquestions = helpers.decompose_question(question, self)
            subanswers = []
            sources = []
            for q in subquestions:
                res = self._ask(q, 
                                qa_template=qa_template, 
                                filters=filters,
                                where_document=where_document,
                                k=k, score_threshold=score_threshold,
                                table_k=table_k, table_score_threshold=table_score_threshold,
                                **kwargs) 
                subanswers.append(res['answer'])
                for doc in res['source_documents']:
                    doc.metadata = dict(doc.metadata, subquestion=q)
                sources.extend(res['source_documents'])
            res = self._ask(question=question,
                            contexts=subanswers,
                            qa_template=qa_template, 
                            filters = filters,
                            where_document=where_document, **kwargs) 
            res['source_documents'] = sources
            return res
        else:       
            res = self._ask(question=question,
                            qa_template=qa_template, 
                            filters = filters,
                            where_document=where_document, **kwargs)
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
