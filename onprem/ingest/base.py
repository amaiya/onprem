"""functionality for text extraction and document ingestion into a vector database for question-answering and other tasks"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/01_ingest.base.ipynb.

# %% auto 0
__all__ = ['logger', 'DEFAULT_CHUNK_SIZE', 'DEFAULT_CHUNK_OVERLAP', 'TABLE_CHUNK_SIZE', 'COLLECTION_NAME', 'CHROMA_MAX',
           'CAPTION_DELIMITER', 'PDFOCR', 'PDFMD', 'PDF', 'PDF_EXTS', 'OCR_CHAR_THRESH', 'LOADER_MAPPING', 'DEFAULT_DB',
           'MyElmLoader', 'MyUnstructuredPDFLoader', 'PDF2MarkdownLoader', 'extract_tables', 'extract_extension',
           'extract_files', 'load_single_document', 'load_documents', 'process_folder', 'process_documents',
           'does_vectorstore_exist', 'batchify_chunks', 'Ingester']

# %% ../../nbs/01_ingest.base.ipynb 3
from ..llm import helpers
from ..utils import split_list, get_datadir, contains_sentence


from langchain_core.documents import Document
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
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings

import os
import os.path
import glob
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
COLLECTION_NAME = "onprem_chroma"
CHROMA_MAX = 41000
CAPTION_DELIMITER = '||CAPTION||'

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
            docs = [Document(page_content=page_content, metadata={'source':source})]
            table_docs = [Document(page_content=t,
                                   metadata={'source':source, 'table':True}) for t in tables]
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
            infer_table_structure = self.text_kwargs.get('infer_table_structure', False)
            if 'infer_table_structure' in self.text_kwargs:
                del self.text_kwargs['infer_table_structure']
            docs = PyMuPDFLoader.load(self)
            if infer_table_structure:
                docs = extract_tables(docs)
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
            md_text = pymupdf4llm.to_markdown(self.file_path)
            if not md_text.strip():
                raise Exception('Document had no content. ')
            doc = Document(page_content=md_text, metadata={'source':self.file_path, 'markdown':True})
            docs = [doc]
            if self.text_kwargs.get('infer_table_structure', False):
                docs = extract_tables(docs)
            return docs
        except Exception as e:
            # Add file_path to exception message
            raise Exception(f'{self.file_path} : {e}')




# %% ../../nbs/01_ingest.base.ipynb 5
# Map file extensions to document loaders and their arguments
PDFOCR = '.pdfOCR'
PDFMD = '.pdfMD'
PDF = '.pdf'
PDF_EXTS = [PDF, PDFOCR, PDFMD]
OCR_CHAR_THRESH = 32
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"autodetect_encoding": True}),
    PDF   : (_PyMuPDFLoader, {}),
    PDFMD: (PDF2MarkdownLoader, {}),
    PDFOCR: (MyUnstructuredPDFLoader, {"infer_table_structure":False, "mode":"elements", "strategy":"hi_res"}),
    # Add more mappings for other file extensions and loaders as needed
}


def extract_tables(docs:List[Document]) -> List[Document]:
    """
    Extract tables from PDF and append to end of Document list.
    """
    from onprem.ingest.pdftables import PDFTables
    filepath = None if not docs else docs[0].metadata['source']
    if not filepath: return docs
    if extract_extension(filepath) != PDF: return docs
    pdftab = PDFTables.from_file(filepath, verbose=False)
    md_tables = pdftab.get_markdown_tables()

    # tag document objects that contain extracted tables
    captions = pdftab.get_captions()
    for c in captions:
        for d in docs:
            if contains_sentence(c, d.page_content):
                table_captions = d.metadata.get('table_captions', [])
                if isinstance(table_captions, str):
                    table_captions = table_captions.split(CAPTION_DELIMITER)
                table_captions.append(c)
                d.metadata['table_captions'] = CAPTION_DELIMITER.join(table_captions)

    # augment docs with extracted tables
    tabledocs = []
    for md_table in md_tables:
        tabledoc = Document(page_content=md_table,
                metadata={'source':filepath, 'markdown':True, 'table':True})
        tabledocs.append(tabledoc)
    docs.extend(tabledocs)
    return docs

def extract_extension(file_path:str):
    """
    Extracts file extension (including dot) from file path
    """
    return "." + file_path.rsplit(".", 1)[-1].lower()


def extract_files(source_dir:str):
    """
    Extract files of supported file types from folder.
    """
    source_dir = os.path.abspath(source_dir)
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.lower()}"), recursive=True)
        )
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.upper()}"), recursive=True)
        )
    return all_files


def load_single_document(file_path: str, # path to file
                         pdf_unstructured:bool=False, # use unstructured for PDF extraction if True (will also OCR if necessary)
                         pdf_markdown:bool = False, # Convert PDFs to Markdown instead of plain text if True.
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
    file_path = os.path.abspath(file_path)
    ext = extract_extension(file_path)
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
                if not docs or len('\n'.join([d.page_content.strip() for d in docs]).strip()) < OCR_CHAR_THRESH:
                    loader_class, loader_args = LOADER_MAPPING[PDFOCR]
                    loader = loader_class(file_path, **loader_args)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata = dict(doc.metadata, ocr=True)
                return docs
            else:
                return loader.load()
        except Exception as e:
            logger.warning(f'\nSkipping {file_path} due to error: {str(e)}')
    else:
        logger.warning(f"\nSkipping {file_path} due to unsupported file extension: '{ext}'")


def load_documents(source_dir: str, # path to folder containing documents
                   ignored_files: List[str] = [], # list of filepaths to ignore
                   ignore_fn:Optional[Callable] = None, # callable that accepts file path and returns True for ignored files
                   caption_tables:bool=False,# If True, agument table text with summaries of tables if infer_table_structure is True.
                   extract_document_titles:bool=False, # If True, infer document title and attach to individual chunks
                   llm=None, # a reference to the LLM (used by `caption_tables` and `extract_document_titles`
                   n_proc:Optional[int]=None, # number of CPU cores to use for text extraction. If None, use maximum for system.
                   **kwargs
) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files.
    Extra kwargs fed to `ingest.load_single_document`.
    """
    all_files = extract_files(source_dir)

    filtered_files = [
        file_path for file_path in all_files if file_path not in ignored_files and not os.path.basename(file_path).startswith('~$')
         and (ignore_fn is None or not ignore_fn(file_path))
    ]

    load_args = kwargs.copy()
    if kwargs.get('infer_table_structure', False):
        # Use "spawn" if using TableTransformers
        # Reference: https://github.com/pytorch/pytorch/issues/40403
        multiprocessing.set_start_method('spawn', force=True)
        # call extract_tables sequentially below instead of in load_single_document if n_proc>1
        # because extract_tables is not well-suited to multiprocessing even with line above
        if not n_proc or n_proc>1:
            load_args = {k:load_args[k] for k in load_args if k!='infer_table_structure'}
    with multiprocessing.Pool(processes=n_proc if n_proc else os.cpu_count()) as pool:
        results = []
        with tqdm(
            total=len(filtered_files), desc="Loading new documents", ncols=80
        ) as pbar:
            for i, docs in enumerate(
                pool.imap_unordered(functools.partial(load_single_document, **load_args),
                                                      filtered_files)
            ):
                if docs is not None:
                    if kwargs.get('infer_table_structure', False):
                        docs = extract_tables(docs)
                    if llm and caption_tables:
                        helpers.caption_tables(docs, llm=llm, **kwargs)
                    if llm and extract_document_titles:
                        title = helpers.extract_title(docs, llm=llm, **kwargs)
                        for doc in docs:
                            if title:
                                doc.metadata['document_title'] = title
                    results.extend(docs)
                pbar.update()

    return results


def process_folder(
    source_directory: str, # path to folder containing document store
    chunk_size: int = DEFAULT_CHUNK_SIZE, # text is split to this many characters by `langchain.text_splitter.RecursiveCharacterTextSplitter`
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP, # character overlap between chunks in `langchain.text_splitter.RecursiveCharacterTextSplitter`
    ignored_files: List[str] = [], # list of files to ignore
    ignore_fn:Optional[Callable] = None, # Callable that accepts the file path (including file name) as input and ignores if returns True
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

    return process_documents(documents,
                             chunk_size = chunk_size,
                             chunk_overlap = chunk_overlap,
                             **kwargs)


def process_documents(
    documents: list, # list of LangChain Documents
    chunk_size: int = DEFAULT_CHUNK_SIZE, # text is split to this many characters by `langchain.text_splitter.RecursiveCharacterTextSplitter`
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP, # character overlap between chunks in `langchain.text_splitter.RecursiveCharacterTextSplitter`
    **kwargs


) -> List[Document]:
    """
    Process list of Documents by splitting into chunks.
    """
    if not documents:
        print("No new documents to process")
        return
    print(f"Processing {len(documents)} new documents")

    # remove tables before chunking
    if kwargs.get('infer_table_structure', False) and not kwargs.get('pdf_unstructured', False):
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
    def includes_caption(d):
        table_captions = d.metadata.get('table_captions', '')
        if not table_captions: return False
        table_captions = table_captions.split(CAPTION_DELIMITER)
        for c in table_captions:
            if contains_sentence(c, d.page_content):
                return True
        return False
    texts = [d for d in texts if not includes_caption(d)]

    # split table texts
    if tables:
        table_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=2000,
            chunk_overlap=0)
        table_texts = table_splitter.split_documents(tables)
        texts.extend(table_texts)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} chars each for text; max. {TABLE_CHUNK_SIZE} chars for tables)")

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


def batchify_chunks(texts):
    """
    split texts into batches specifically for Chroma
    """
    split_docs_chunked = split_list(texts, CHROMA_MAX)
    total_chunks = sum(1 for _ in split_list(texts, CHROMA_MAX))
    return split_docs_chunked, total_chunks



os.environ["TOKENIZERS_PARALLELISM"] = "0"
DEFAULT_DB = "vectordb"


class Ingester:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_model_kwargs: dict = {"device": "cpu"},
        embedding_encode_kwargs: dict = {"normalize_embeddings": False},
        persist_directory: Optional[str] = None,
    ):
        """
        Ingests all documents in `source_folder` (previously-ingested documents are ignored)

        **Args**:

          - *embedding_model*: name of sentence-transformers model
          - *embedding_model_kwargs*: arguments to embedding model (e.g., `{device':'cpu'}`)
          - *embedding_encode_kwargs*: arguments to encode method of
                                       embedding model (e.g., `{'normalize_embeddings': False}`).
          - *persist_directory*: Path to vector database (created if it doesn't exist).
                                 Default is `onprem_data/vectordb` in user's home directory.


        **Returns**: `None`
        """
        self.persist_directory = persist_directory or os.path.join(
            get_datadir(), DEFAULT_DB
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=embedding_model_kwargs,
            encode_kwargs=embedding_encode_kwargs,
        )
        self.chroma_settings = Settings(
            persist_directory=self.persist_directory, anonymized_telemetry=False
        )
        self.chroma_client = chromadb.PersistentClient(
            settings=self.chroma_settings, path=self.persist_directory
        )
        return

    def get_db(self):
        """
        Returns an instance to the `langchain_chroma.Chroma` instance
        """
        db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            client_settings=self.chroma_settings,
            client=self.chroma_client,
            collection_metadata={"hnsw:space": "cosine"},
            collection_name=COLLECTION_NAME,
        )
        return db if does_vectorstore_exist(db) else None

    def get_embedding_model(self):
        """
        Returns an instance to the `langchain_huggingface.HuggingFaceEmbeddings` instance
        """
        return self.embeddings


    def get_ingested_files(self):
        """
        Returns a list of files previously added to vector database (typically via `LLM.ingest`)
        """
        return set([d['source'] for d in self.get_db().get()['metadatas']])


    def store_documents(self, documents):
        """
        Stores instances of `langchain_core.documents.base.Document` in vectordb
        """
        if not documents:
            return
        db = self.get_db()
        if db:
            print("Creating embeddings. May take some minutes...")
            chunk_batches, total_chunks = batchify_chunks(documents)
            for lst in tqdm(chunk_batches, total=total_chunks):
                db.add_documents(lst)
        else:
            chunk_batches, total_chunks = batchify_chunks(documents)
            print("Creating embeddings. May take some minutes...")
            db = None

            for lst in tqdm(chunk_batches, total=total_chunks):
                if not db:
                    db = Chroma.from_documents(
                        lst,
                        self.embeddings,
                        persist_directory=self.persist_directory,
                        client_settings=self.chroma_settings,
                        client=self.chroma_client,
                        collection_metadata={"hnsw:space": "cosine"},
                        collection_name=COLLECTION_NAME,
                    )
                else:
                    db.add_documents(lst)
        return


    def ingest(
        self,
        source_directory: str, # path to folder containing document store
        chunk_size: int = DEFAULT_CHUNK_SIZE, # text is split to this many characters by [langchain.text_splitter.RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP, # character overlap between chunks in `langchain.text_splitter.RecursiveCharacterTextSplitter`
        ignore_fn:Optional[Callable] = None, # Optional function that accepts the file path (including file name) as input and returns `True` if file path should not be ingested.
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
        db = self.get_db()
        if db:
            # Update and store locally vectorstore
            print(f"Appending to existing vectorstore at {self.persist_directory}")
            collection = db.get()
            ignored_files=[ metadata["source"] for metadata in collection["metadatas"]]
        else:
            print(f"Creating new vectorstore at {self.persist_directory}")
            ignored_files = []

        texts = process_folder(
            source_directory,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            ignored_files=ignored_files,
            ignore_fn=ignore_fn,
            **kwargs

        )
        self.store_documents(texts)

        if texts:
            print(
                "Ingestion complete! You can now query your documents using the LLM.ask or LLM.chat methods"
            )
        db = None
        return
