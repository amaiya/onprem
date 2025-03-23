"""full-text search engine"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/01_ingest.stores.sparse.ipynb.

# %% auto 0
__all__ = ['DEFAULT_SCHEMA', 'default_schema', 'SparseStore']

# %% ../../../nbs/01_ingest.stores.sparse.ipynb 3
import json
import os
import warnings
from typing import Dict, List, Optional, Sequence
import math

from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import *
from whoosh.filedb.filestore import RamStorage
from whoosh.qparser import MultifieldParser
from langchain_core.documents import Document
import uuid
from tqdm import tqdm
from ..base import VectorStore

# ------------------------------------------------------------------------------
# IMPORTANT: Metadata fields in langchain_core.documents.Document objects
#            (i.e., the input to WSearch.index_documents) should
#            ideally match schema fields below, but this is not strictly required.
#
#            The page_content field is the only truly required field in supplied
#            Document objects. All other fields, including dynamic fields, are optional. 
# ------------------------------------------------------------------------------

DEFAULT_SCHEMA = Schema(
    page_content=TEXT(stored=True), # REQUIRED
    id=ID(stored=True, unique=True),
    source=KEYWORD(stored=True, commas=True), 
    source_search=TEXT(stored=True),
    filepath=KEYWORD(stored=True, commas=True),
    filepath_search=TEXT(stored=True),
    filename=KEYWORD(stored=True),
    ocr=BOOLEAN(stored=True),
    table=BOOLEAN(stored=True),
    markdown=BOOLEAN(stored=True),
    page=NUMERIC(stored=True),
    document_title=TEXT(stored=True),
    md5=KEYWORD(stored=True),
    mimetype=KEYWORD(stored=True),
    extension=KEYWORD(stored=True),
    filesize=NUMERIC(stored=True),
    createdate=DATETIME(stored=True),
    modifydate=DATETIME(stored=True),
    tags=KEYWORD(stored=True, commas=True),
    notes=TEXT(stored=True),
    msg=TEXT(stored=True),
    )
DEFAULT_SCHEMA.add("*_t", TEXT(stored=True), glob=True)
DEFAULT_SCHEMA.add("*_k", KEYWORD(stored=True, commas=True), glob=True)
DEFAULT_SCHEMA.add("*_b", BOOLEAN(stored=True), glob=True)
DEFAULT_SCHEMA.add("*_n", NUMERIC(stored=True), glob=True)
DEFAULT_SCHEMA.add("*_d", DATETIME(stored=True), glob=True)


def default_schema():
    schema = DEFAULT_SCHEMA
    #if "raw" not in schema.stored_names():
        #schema.add("raw", TEXT(stored=True))
    return schema


class SparseStore(VectorStore):
    def __init__(self,
                persist_directory: Optional[str]=None, # path to folder where search index is stored
                index_name:str = 'myindex',            # name of index
                **kwargs,
        ):
        """
        Initializes full-text search engine
        """

        self.index_path = persist_directory
        self.index_name = index_name
        if self.index_path and not self.index_name:
            raise ValueError('index_name is required if index_path is supplied')
        if self.index_path:
            if not index.exists_in(self.index_path, indexname=self.index_name):
                self.ix = __class__.initialize_index(self.index_path, self.index_name)
            else:
                self.ix = index.open_dir(self.index_path, indexname=self.index_name)
        else:
            warnings.warn(
                "No persist_directory was supplied, so an in-memory only index"
                "was created using DEFAULT_SCHEMA"
            )
            self.ix = RamStorage().create_index(default_schema())


    def get_db(self):
        """
        Get raw index
        """
        return self.ix


    def exists(self):
        """
        Returns True if documents have been added to search index
        """
        return self.get_size() > 0


    def add_documents(self,
                      docs: Sequence[Document], # list of LangChain Documents
                      limitmb:int=1024, # maximum memory in  megabytes to use
                      verbose:bool=True, # Set to False to disable progress bar
                      **kwargs,
        ):
        """
        Indexes documents. Extra kwargs supplied to `TextStore.ix.writer`.
        """
        writer = self.ix.writer(limitmb=limitmb, **kwargs)
        for doc in tqdm(docs, total=len(docs), disable=not verbose):
            d = self.doc2dict(doc)
            writer.update_document(**d)
        writer.commit(optimize=True)


    def remove_document(self, value:str, field:str='id'):
        """
        Remove document with corresponding value and field.
        Default field is the id field.
        """
        writer = self.ix.writer()
        writer.delete_by_term(field, value)
        writer.commit(optimize=True)
        return



    def get_all_docs(self):
        """
        Returns a generator to iterate through all indexed documents
        """
        return self.ix.searcher().documents()
        

    def get_doc(self, id:str):
        """
        Get an indexed record by ID
        """
        r = self.query(f'id:{id}')
        return r['hits'][0] if len(r['hits']) > 0 else None


    def get_size(self) -> int:
        """
        Gets size of index
        """
        return self.ix.doc_count_all()

        
    def erase(self, confirm=True):
        """
        Clears index
        """
        shall = True
        if confirm:
            msg = (
                f"You are about to remove all documents from the search index."
                + f"(Original documents on file system will remain.) Are you sure?"
            )
            shall = input("%s (Y/n) " % msg) == "Y"
        if shall and index.exists_in(
            self.index_path, indexname=self.index_name
        ):
            ix = index.create_in(
                self.index_path,
                indexname=self.index_name,
                schema=default_schema(),
            )
            return True
        return False


    def query(
            self,
            q: str,
            fields: Sequence = ["page_content"],
            highlight: bool = True,
            limit:int=10,
            page:int=1,
    ) -> List[Dict]:
        """
        Queries the index

        **Args**

        - *q*: the query string
        - *fields*: a list of fields to search
        - *highlight*: If True, highlight hits
        - *limit*: results per page
        - *page*: page of hits to return
        """
        search_results = []
        with self.ix.searcher() as searcher:
            if page == 1:
                results = searcher.search(
                    MultifieldParser(fields, schema=self.ix.schema).parse(q), limit=limit)
            else:
                results = searcher.search_page(
                    MultifieldParser(fields, schema=self.ix.schema).parse(q), page, limit)
            total_hits = results.scored_length()
            if page > math.ceil(total_hits/limit):
               results = []
            for r in results:
                #d = json.loads(r["raw"])
                d = dict(r)
                if highlight:
                    for f in fields:
                        if r[f] and isinstance(r[f], str):
                            d['hl_'+f] = r.highlights(f) or r[f]

                search_results.append(d)

        return {'hits':search_results, 'total_hits':total_hits}

    def semantic_search(self, *args, **kwargs):
        """
        Not yet implemented
        """
        raise NotImplementedError('This method has not yet been implemented for SparseStore.')
        
    @classmethod
    def index_exists_in(cls, index_path: str, index_name: Optional[str] = None):
        """
        Returns True if index exists with name, *indexname*, and path, *index_path*.
        """
        return index.exists_in(index_path, indexname=index_name)

    @classmethod
    def initialize_index(
        cls, index_path: str, index_name: str, schema: Optional[Schema] = None
    ):
        """
        Initialize index

        **Args**

        - *index_path*: path to folder storing search index
        - *index_name*: name of index
        - *schema*: optional whoosh.fields.Schema object.
                    If None, DEFAULT_SCHEMA is used
        """
        schema = default_schema() if not schema else schema

        if index.exists_in(index_path, indexname=index_name):
            raise ValueError(
                f"There is already an existing index named {index_name}  with path {index_path} \n"
                + f"Delete {index_path} manually and try again."
            )
        if not os.path.exists(index_path):
            os.makedirs(index_path)
        ix = index.create_in(index_path, indexname=index_name, schema=schema)
        return ix

    def doc2dict(self, doc:Document):
        """
        Convert LangChain Document to expected format
        """
        stored_names = self.ix.schema.stored_names()
        d = {}
        for k,v in doc.metadata.items():
            suffix = None
            if k in stored_names:
                suffix = ''
            elif isinstance(v, bool):
                suffix = '_b' if not k.endswith('_b') else ''
            elif isinstance(v, str):
                if k.endswith('_date'):
                    suffix = '_d'
                else:
                    suffix = '_k'if not k.endswith('_k') else ''
            elif isinstance(v, (int, float)):
                suffix = '_n'if not k.endswith('_n') else ''
            if suffix is not None:
                d[k+suffix] = v
        d['id'] = uuid.uuid4().hex
        d['page_content' ] = doc.page_content
        #d['raw'] = json.dumps(d)
        if 'source' in d:
            d['source_search'] = d['source']
        if 'filepath' in d:
            d['filepath_search'] = d['filepath']
        return d





