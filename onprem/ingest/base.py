"""functionality for text extraction and document ingestion into a vector database for question-answering and other tasks"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/01_ingest.base.ipynb.

# %% auto 0
__all__ = ['logger', 'DEFAULT_CHUNK_SIZE', 'DEFAULT_CHUNK_OVERLAP', 'TABLE_CHUNK_SIZE', 'CHROMA_MAX', 'PDFOCR', 'PDFMD', 'PDF',
           'PDF_EXTS', 'OCR_CHAR_THRESH', 'LOADER_MAPPING', 'MyElmLoader', 'MyUnstructuredPDFLoader',
           'PDF2MarkdownLoader', 'load_single_document', 'load_documents', 'process_folder', 'chunk_documents',
           'does_vectorstore_exist', 'batchify_chunks', 'VectorStore']

# %% ../../nbs/01_ingest.base.ipynb 3
from ..llm.helpers import summarize_tables, extract_title
from ..utils import batch_list, filtered_generator
from . import helpers

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import Language
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    TextLoader,
    PyMuPDFLoader,
    UnstructuredPDFLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)


import os
import os.path
from typing import List, Optional, Callable
import multiprocessing
import functools
from tqdm import tqdm
import warnings
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging.getLogger('OnPrem.LLM-ingest')

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
TABLE_CHUNK_SIZE = 2000
CHROMA_MAX = 41000

# %% ../../nbs/01_ingest.base.ipynb 4
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if "text/html content not found in email" in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            raise Exception(f'{self.file_path} : {e}')

        return doc


class MyUnstructuredPDFLoader(UnstructuredPDFLoader):
    """Custom PDF Loader"""

    def load(self) -> List[Document]:
        """Wrapper UnstructuredPDFLoader"""
        try:
            docs = UnstructuredPDFLoader.load(self)
            if not docs:
                raise Exception('Document had no content. ')
            tables = [d.metadata['text_as_html'] for d in docs if d.metadata.get('text_as_html', None) is not None]
            texts = [d.page_content for d in docs if d.metadata.get('text_as_html', None) is None]

            page_content = '\n'.join(texts)
            source = docs[0].metadata['source']
            docs = [helpers.create_document(page_content, source)]
            table_docs = [helpers.create_document(t, source=source, table=True) for t in tables]
            docs.extend(table_docs)
            return docs
        except Exception as e:
            # Add file_path to exception message
            raise Exception(f'{self.file_path} : {e}')


class _PyMuPDFLoader(PyMuPDFLoader):
    """Custom PyMUPDF Loader with optional support for inferring table structure"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            # PyMuPDFLoader complains when you add custom flags to text_kwargs,
            # so delete before loading
            infer_table_structure = self.parser.text_kwargs.get('infer_table_structure', False)
            if 'infer_table_structure' in self.parser.text_kwargs:
                del self.parser.text_kwargs['infer_table_structure']
            docs = PyMuPDFLoader.load(self)
            if infer_table_structure:
                docs = helpers.extract_tables(docs=docs)
            return docs
        except Exception as e:
            # Add file_path to exception message
            raise Exception(f'{self.file_path} : {e}')


class PDF2MarkdownLoader(_PyMuPDFLoader):
    """Custom PDF to Markdown Loader"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        import pymupdf4llm
        try:
            md_text = pymupdf4llm.to_markdown(self.file_path, show_progress=False)
            if not md_text.strip():
                raise Exception('Document had no content. ')
            doc = helpers.create_document(md_text, self.file_path, markdown=True)
            docs = [doc]
            if self.parser.text_kwargs.get('infer_table_structure', False):
                docs = helpers.extract_tables(docs=docs)
            return docs
        except Exception as e:
            # Add file_path to exception message
            raise Exception(f'{self.file_path} : {e}')




# %% ../../nbs/01_ingest.base.ipynb 5
# Map file extensions to document loaders and their arguments
PDFOCR = 'pdfOCR'
PDFMD = 'pdfMD'
PDF = 'pdf'
PDF_EXTS = [PDF, PDFOCR, PDFMD]
OCR_CHAR_THRESH = 32
LOADER_MAPPING = {
    "csv": (CSVLoader, {}),
    "doc": (UnstructuredWordDocumentLoader, {}),
    "docx": (UnstructuredWordDocumentLoader, {}),
    "enex": (EverNoteLoader, {}),
    "eml": (MyElmLoader, {}),
    "epub": (UnstructuredEPubLoader, {}),
    "html": (UnstructuredHTMLLoader, {}),
    "md": (UnstructuredMarkdownLoader, {}),
    "odt": (UnstructuredODTLoader, {}),
    "ppt": (UnstructuredPowerPointLoader, {}),
    "pptx": (UnstructuredPowerPointLoader, {}),
    "txt": (TextLoader, {"autodetect_encoding": True}),
    PDF   : (_PyMuPDFLoader, {}),
    PDFMD: (PDF2MarkdownLoader, {}),
    PDFOCR: (MyUnstructuredPDFLoader, {"infer_table_structure":False, "mode":"elements", "strategy":"hi_res"}),
    # Add more mappings for other file extensions and loaders as needed
}

def _update_metadata(docs:List[Document], metadata:dict):
    """
    Update metadata in docs with supplied metadata dictionary
    """
    for doc in docs:
        doc.metadata.update(metadata)
    return docs

def _apply_text_callables(docs:List[Document], text_callables:dict):
    """
    Invokes text_callables on entire text of document.

    Returns a dictionary with values containing results from callables for each key
    """
    if not text_callables: return {}
        
    text = '\n\n'.join([d.page_content for d in docs])
    results = {}
    for k,v in text_callables.items():
        results[k] = v(text)
    return results

def _apply_file_callables(file_path:str, file_callables:dict):
    """
    Invokes file_callables on file path.

    Returns a dictionary with values containing results from callables for each key
    """
    if not file_callables: return {}
        
    if not os.path.exists(file_path):
        raise ValueError('file_path does not exist: {file_path}')
    
    results = {}
    for k,v in file_callables.items():
        results[k] = v(file_path)
    return results

    
def load_single_document(file_path: str, # path to file
                         pdf_unstructured:bool=False, # use unstructured for PDF extraction if True (will also OCR if necessary)
                         pdf_markdown:bool = False, # Convert PDFs to Markdown instead of plain text if True.
                         store_md5:bool=False, # Extract and store MD5 of document in metadata
                         store_mimetype:bool=False, # Guess and store mime type of document in metadata
                         store_file_dates:bool=False, # Extract snd store file dates in metadata
                         file_callables:Optional[dict]=None, # optional dict with  keys and functions called with filepath as argument. Results stored as metadata.
                         text_callables:Optional[dict]=None, # optional dict with  keys and functions called with file text as argument. Results stored as metadata.
                         **kwargs,
                         ) -> List[Document]:
    """
    Extract text from a single document. Will attempt to OCR PDFs, if necessary.


    Note that extra kwargs can be supplied to configure the behavior of PDF loaders.
    For instance, supplying `infer_table_structure` will cause `load_single_document` to try and
    infer and extract tables from PDFs. When `pdf_unstructured=True` and `infer_table_structure=True`,
    tables are represented as HTML within the main body of extracted text. In all other cases, inferred tables
    are represented as Markdown and appended to the end of the extracted text when `infer_table_structure=True`.
    """
    if pdf_unstructured and pdf_markdown:
        raise ValueError('pdf_unstructured and pdf_markdown cannot both be True.')
    file_callables = {} if not file_callables else file_callables
    text_callables = {} if not text_callables else text_callables
    file_path = os.path.abspath(file_path)


    # extract metadata
    file_metadata = {}
    if store_md5:
        file_metadata['md5'] = helpers.md5sum(file_path)
    if store_mimetype:
        file_metadata['mimetype'], _, _ = helpers.extract_mimetype(file_path)
    if store_file_dates:
        file_metadata['createdate'], file_metadata['modifydate'] = helpers.extract_file_dates(file_path)
    ext = helpers.extract_extension(file_path)
    file_metadata['extension'] = ext
    file_metadata.update(_apply_file_callables(file_path, file_callables))
        
    # load file
    if ext in LOADER_MAPPING:
        try:
            if ext == PDF:
                if pdf_unstructured:
                    ext = PDFOCR
                elif pdf_markdown:
                    ext = PDFMD
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader_args = loader_args.copy() # copy so any supplied kwargs do not persist across calls
            if ext in PDF_EXTS:
                loader_args.update(kwargs)
            loader = loader_class(file_path, **loader_args)
            if ext in PDF_EXTS and ext != PDFOCR:
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=UserWarning)
                    docs = loader.load()
                    file_metadata.update(_apply_text_callables(docs, text_callables))
                    docs = _update_metadata(docs, file_metadata)
                if not docs or len('\n'.join([d.page_content.strip() for d in docs]).strip()) < OCR_CHAR_THRESH:
                    loader_class, loader_args = LOADER_MAPPING[PDFOCR]
                    loader = loader_class(file_path, **loader_args)
                    file_metadata['ocr'] = True
                    docs = loader.load()
                    file_metadata.update(_apply_text_callables(docs, text_callables))
                    docs = _update_metadata(docs, file_metadata)
            else:
                docs = loader.load()
                file_metadata.update(_apply_text_callables(docs, text_callables))                
                docs = _update_metadata(docs, file_metadata)
            extra_keys = list(file_metadata.keys() | text_callables.keys())
            return helpers.set_metadata_defaults(docs, extra_keys=extra_keys)
        except Exception as e:
            logger.warning(f'\nSkipping {file_path} due to error: {str(e)}')
            import traceback
            print(traceback.format_exc())

    else:
        logger.warning(f"\nSkipping {file_path} due to unsupported file extension: '{ext}'")


def _ignore_file(file_path, ignored_files:List[str]=[], ignore_fn:Optional[Callable]=None):
    file_path = os.path.abspath(file_path)
    return file_path in ignored_files or \
            os.path.basename(file_path).startswith('~$') or \
            (ignore_fn is not None and ignore_fn(file_path))


def load_documents(source_dir: str, # path to folder containing documents
                   ignored_files: List[str] = [], # list of filepaths to ignore
                   ignore_fn:Optional[Callable] = None, # callable that accepts file path and returns True for ignored files
                   caption_tables:bool=False,# If True, agument table text with summaries of tables if infer_table_structure is True.
                   extract_document_titles:bool=False, # If True, infer document title and attach to individual chunks
                   llm=None, # a reference to the LLM (used by `caption_tables` and `extract_document_titles`
                   n_proc:Optional[int]=None, # number of CPU cores to use for text extraction. If None, use maximum for system.
                   verbose:bool=True, # verbosity
                   **kwargs
) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files.
    Extra kwargs fed to `ingest.load_single_document`.

    Returns a generator over documents.
    """

    def keep(file_path):
        return not _ignore_file(file_path, ignored_files, ignore_fn)
    def all_files():
        for f in helpers.extract_files(source_dir, LOADER_MAPPING):
            yield f
    filtered_files = filtered_generator(all_files(), criteria=[keep])
    total = sum(1 for _ in filtered_generator(all_files(), criteria=[keep]))

    load_args = kwargs.copy()
    if kwargs.get('infer_table_structure', False):
        # Use "spawn" if using TableTransformers
        # Reference: https://github.com/pytorch/pytorch/issues/40403
        multiprocessing.set_start_method('spawn', force=True)
        # call helpers.extract_tables sequentially below instead of in load_single_document if n_proc>1
        # because helpers.extract_tables is not well-suited to multiprocessing even with line above
        if not n_proc or n_proc>1:
            load_args = {k:load_args[k] for k in load_args if k!='infer_table_structure'}
    with multiprocessing.Pool(processes=n_proc if n_proc else os.cpu_count()) as pool:
        results = []
        with tqdm(
            total=total, desc="Loading new documents", ncols=80, disable=not verbose
        ) as pbar:
            for i, docs in enumerate(
                pool.imap_unordered(functools.partial(load_single_document, **load_args),
                                                      filtered_files)
            ):
                pbar.update()
                if docs is None: continue
                if kwargs.get('infer_table_structure', False):
                    docs = helpers.extract_tables(docs=docs)
                if llm and caption_tables:
                    summarize_tables(docs, llm=llm, **kwargs)
                if llm and extract_document_titles:
                    title = extract_title(docs, llm=llm, **kwargs)
                    for doc in docs:
                        if title:
                            doc.metadata['document_title'] = title
                yield from docs


def process_folder(
    source_directory: str, # path to folder containing document store
    chunk_size: int = DEFAULT_CHUNK_SIZE, # text is split to this many characters by `langchain.text_splitter.RecursiveCharacterTextSplitter`
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP, # character overlap between chunks in `langchain.text_splitter.RecursiveCharacterTextSplitter`
    ignored_files: List[str] = [], # list of files to ignore
    ignore_fn:Optional[Callable] = None, # Callable that accepts the file path (including file name) as input and ignores if returns True
    batch_size:int=CHROMA_MAX, # batch size used when processing documents
    **kwargs


) -> List[Document]:
    """
    Load documents from folder, extract text from them, split texts into chunks.
    Extra kwargs fed to `ingest.load_documents` and `ingest.load_single_document`.
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory,
                              ignored_files, ignore_fn=ignore_fn,
                              **kwargs)
    documents = list(documents)

    if not documents:
        print("No new documents to process")
        return

    batches = batch_list(documents, batch_size)
    total = sum(1 for _ in batch_list(documents, batch_size))
    for docs in tqdm(batches, total=total,
                     desc=f'Processing and chunking {len(documents)} new documents'):
        yield from chunk_documents(docs,
                                  chunk_size = chunk_size,
                                  chunk_overlap = chunk_overlap,
                                  **kwargs)


def chunk_documents(
    documents: list, # list of LangChain Documents
    chunk_size: int = DEFAULT_CHUNK_SIZE, # text is split to this many characters by `langchain.text_splitter.RecursiveCharacterTextSplitter`
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP, # character overlap between chunks in `langchain.text_splitter.RecursiveCharacterTextSplitter`
    infer_table_structure:bool = False, # This should be set to True if `documents` may contain contain tables (i.e., `doc.metadata['table']=True`).
    **kwargs


) -> List[Document]:
    """
    Process list of Documents by splitting into chunks.
    """
    # remove tables before chunking
    if infer_table_structure and not kwargs.get('pdf_unstructured', False):
        tables = [d for d in documents if d.metadata.get('table', False)]
        docs = [d for d in documents if not d.metadata.get('table', False)]
    else:
        tables = []
        docs = documents

    # initialize the splitter
    contains_markdown = kwargs.get('pdf_markdown', False)
    if contains_markdown:
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap)
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    # split non-table texts
    texts = text_splitter.split_documents(docs)

    # attempt to remove text chunks containing mangled tables
    texts = [d for d in texts if not helpers.includes_caption(d)]

    # split table texts
    if tables:
        table_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=2000,
            chunk_overlap=0)
        table_texts = table_splitter.split_documents(tables)
        texts.extend(table_texts)

    # attach document title to each chunk (where title was extracted earlier by `load_documents`)
    if kwargs.get('extract_document_titles', False):
        for text in texts:
            if text.metadata.get('document_title', ''):
                text.page_content = f'The content below is from a document titled, \"{text.metadata["document_title"]}\"\n\n{text.page_content}'
    return texts


def does_vectorstore_exist(db) -> bool:
    """
    Checks if vectorstore exists
    """
    if not db.get()["documents"]:
        return False
    return True


def batchify_chunks(texts, batch_size=CHROMA_MAX):
    """
    split texts into batches specifically for Chroma
    """
    split_docs_chunked = batch_list(texts, batch_size)
    total_chunks = sum(1 for _ in batch_list(texts, batch_size))
    return split_docs_chunked, total_chunks





# %% ../../nbs/01_ingest.base.ipynb 6
from abc import ABC, abstractmethod

class VectorStore(ABC):

    def ingest(
        self,
        source_directory: str, # path to folder containing document store
        chunk_size: int = DEFAULT_CHUNK_SIZE, # text is split to this many characters by [langchain.text_splitter.RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP, # character overlap between chunks in `langchain.text_splitter.RecursiveCharacterTextSplitter`
        ignore_fn:Optional[Callable] = None, # Optional function that accepts the file path (including file name) as input and returns `True` if file path should not be ingested.
        batch_size:int=CHROMA_MAX, # batch size used when processing documents
        **kwargs
    ) -> None:
        """
        Ingests all documents in `source_directory` (previously-ingested documents are
        ignored). When retrieved, the
        [Document](https://api.python.langchain.com/en/latest/documents/langchain_core.documents.base.Document.html)
        objects will each have a `metadata` dict with the absolute path to the file
        in `metadata["source"]`.
        Extra kwargs fed to `ingest.load_single_document`.
        """

        if not os.path.exists(source_directory):
            raise ValueError("The source_directory does not exist.")
        elif os.path.isfile(source_directory):
            raise ValueError(
                "The source_directory argument must be a folder, not a file."
            )
        texts = None
        if self.exists():
            # Update and store locally vectorstore
            print(f"Appending to existing vectorstore at {self.persist_directory}")
            ignored_files = set([d['source'] for d in self.get_all_docs()])
        else:
            print(f"Creating new vectorstore at {self.persist_directory}")
            ignored_files = []

        texts = process_folder(
            source_directory,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            ignored_files=ignored_files,
            ignore_fn=ignore_fn,
            batch_size=batch_size,
            **kwargs

        )

        texts = list(texts)
        print(f"Split into {len(texts)} chunks of text (max. {chunk_size} chars each for text; max. {TABLE_CHUNK_SIZE} chars for tables)")

        self.add_documents(texts, batch_size=batch_size)

        if texts:
            print(
                "Ingestion complete! You can now query your documents using the LLM.ask or LLM.chat methods"
            )
        db = None
        return
    
    def init_embedding_model(self, 
                             embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                             embedding_model_kwargs: Optional[dict] = None,
                             embedding_encode_kwargs: dict = {"normalize_embeddings": False},
                             **kwargs
                             ):
        """
        Instantiate embedding model
        """
        if not embedding_model_kwargs:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            embedding_model_kwargs = {"device": device}          
        self.embeddings =  HuggingFaceEmbeddings(model_name=embedding_model_name, 
                                     model_kwargs=embedding_model_kwargs,
                                     encode_kwargs=embedding_encode_kwargs)

    def get_embedding_model(self):
        """
        Returns an instance to the `langchain_huggingface.HuggingFaceEmbeddings` instance
        """
        return self.embeddings


    def check(self):
        """
        Raise exception if `VectorStore.exists()` returns False
        """
        if not self.exists():
            raise Exception('The vector store is either empty or does not yet exist. '+\
                            'Please invoke the `ingest` method or `add_document` method.')

    @abstractmethod
    def get_db(self):
        """
        Get the raw, underlying vector database or search index.
        """
        pass


    @abstractmethod
    def exists(self):
        """
        Returns True if vector store has been initialized and contains documents.
        """
        pass


    @abstractmethod
    def add_documents(self, documents):
        """
        Stores instances of `langchain_core.documents.base.Document` in vector store
        """
        pass

    @abstractmethod
    def remove_document(self, id_to_delete):
        """
        Remove a single document.
        """
        pass
    
    @abstractmethod
    def update_documents(self, *args, **kwargs):
        """
        Update a set of documents.
        """
        pass

    @abstractmethod
    def get_all_docs(self):
        """
        Returns a list of files previously added to vector store.
        """
        pass

    @abstractmethod
    def get_doc(self, id):
        """
        Retrieve a document by ID
        """
        pass

    @abstractmethod
    def get_size(self):
        """
        Get total number of records added to vector store
        """
        pass

    @abstractmethod
    def erase(self):
        """
        Removes all documents in vector store
        """
        pass

    @abstractmethod
    def query(self, query):
        """
        Queries the vector store.
        For sparse stores, this is simply a keyword-search.
        For dense stores, this is equivalent to semantic_search.
        """
        pass

    @abstractmethod
    def semantic_search(self):
        """
        Semantic search of vector store
        """
        pass
