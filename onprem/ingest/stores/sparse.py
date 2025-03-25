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
import numpy as np

from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import *
from whoosh.filedb.filestore import RamStorage
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh.query import Term, And
from langchain_core.documents import Document
import uuid
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from ..base import VectorStore
from ..helpers import doc_from_dict

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
                persist_directory: Optional[str]=None, 
                index_name:str = 'myindex',
                **kwargs,
        ):
        """
        Initializes full-text search engine.

        **Args:**

        - *persist_directory*: path to folder where search index is stored
        - *index_name*: name of index
        - *embedding_model*: name of sentence-transformers model
        - *embedding_model_kwargs*: arguments to embedding model (e.g., `{device':'cpu'}`). If None, GPU used if available.
        - *embedding_encode_kwargs*: arguments to encode method of
                                     embedding model (e.g., `{'normalize_embeddings': False}`).
        """

        self.persist_directory = persist_directory # alias for consistency with DenseStore
        self.index_name = index_name
        if self.persist_directory and not self.index_name:
            raise ValueError('index_name is required if persist_directory is supplied')
        if self.persist_directory:
            if not index.exists_in(self.persist_directory, indexname=self.index_name):
                self.ix = __class__.initialize_index(self.persist_directory, self.index_name)
            else:
                self.ix = index.open_dir(self.persist_directory, indexname=self.index_name)
        else:
            warnings.warn(
                "No persist_directory was supplied, so an in-memory only index"
                "was created using DEFAULT_SCHEMA"
            )
            self.ix = RamStorage().create_index(default_schema())
        self.init_embedding_model(**kwargs) # stored as self.embeddings

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


    def update_documents(self,
                         doc_dicts:dict, # dictionary with keys 'page_content', 'source', 'id', etc.
                         **kwargs):
        """
        Update a set of documents (doc in index with same ID will be over-written)
        """
        docs = [doc_from_dict(d) for d in doc_dicts]
        self.add_documents(docs)


    def get_all_docs(self):
        """
        Returns a generator to iterate through all indexed documents
        """
        return self.ix.searcher().documents()
       

    def get_doc(self, id:str):
        """
        Get an indexed record by ID
        """
        r = self.query(f'id:{id}', return_dict=True)
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
            self.persist_directory, indexname=self.index_name
        ):
            ix = index.create_in(
                self.persist_directory,
                indexname=self.index_name,
                schema=default_schema(),
            )
            return True
        return False

    def _analyze_query(self, q, field:str='page_content'):
        """
        Analyze query
        """
        analyzer = self.ix.schema[field].analyzer
        return " ".join([token.text for token in analyzer(q)])

    
    def query(
            self,
            q: str,
            fields: Sequence = ["page_content"],
            highlight: bool = True,
            limit:int=10,
            page:int=1,
            return_dict:bool=False,
            filters:Optional[Dict[str, str]] = None,
            where_document:Optional[str]=None,

    ) -> List[Dict]:
        """
        Queries the index

        **Args**

        - *q*: the query string
        - *fields*: a list of fields to search
        - *highlight*: If True, highlight hits
        - *limit*: results per page
        - *page*: page of hits to return
        - *return_dict*: If True, return list of dictionaries instead of LangChain Document objects
        - *filters*: filter results by field values (e.g., {'extension':'pdf'})
        - *where_document*: optional query to further filter results
        """

        search_results = []

        # Apply analyzer to query as long as it is not a boolean query
        if "AND" not in q and "OR" not in q and "NOT" not in q and ":" not in q:
            q = self._analyze_query(q)
        if where_document:
            q = f'({q}) AND ({where_document})'

        # process filters
        combined_filter=None
        if filters:
            terms = []
            for k in filters:
                terms.append(Term(k, filters[k]))
            combined_filter = And(terms)
                   
        # process search
        with self.ix.searcher() as searcher:
            if page == 1:
                results = searcher.search(
                    MultifieldParser(fields, schema=self.ix.schema, group=OrGroup.factory(0.9)).parse(q), limit=limit, filter=combined_filter)
            else:
                results = searcher.search_page(
                    MultifieldParser(fields, schema=self.ix.schema, group=OrGroup.factory(0.9)).parse(q), page, limit, filter=combined_filter)
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
                d = d if return_dict else doc_from_dict(d)
                search_results.append(d)
   
        return {'hits':search_results, 'total_hits':total_hits}

    def semantic_search(self, 
                        query, 
                        k:int=4, # number of records to return based on highest semantic similarity scores.
                        n_candidates=50, # Number of records to consider (for which we compute embeddings on-the-fly)
                        filters:Optional[Dict[str, str]] = None, # filter sources by field values (e.g., {'table':True})
                        where_document:Optional[str]=None, # a boolean query to filter results further (e.g., "climate" AND extension:pdf)
                        **kwargs):
        """
        Retrieves results based on semantic similarity to supplied `query`.
        """
        from sentence_transformers import util
        import torch

        results = self.query(query, limit=n_candidates, return_dict=True, filters=filters, where_document=where_document)['hits']
        if not results: return []
        texts = [r['page_content'] for r in results]
        embeddings = self.get_embedding_model()

        # Compute embeddings
        query_emb = torch.tensor(embeddings.embed_query(query)).unsqueeze(0)  # Shape (1, embedding_dim)
        text_embs = torch.tensor(embeddings.embed_documents(texts))  # Shape (len(texts), embedding_dim)
    
        # Compute cosine similarity
        cos_scores = util.pytorch_cos_sim(query_emb, text_embs).squeeze(0).tolist()  # Shape (len(texts),)

        # Assign scores back to results
        for i, score in enumerate(cos_scores):
            results[i]['score'] = score

        # Sort results by similarity in descending order
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)[:k]
        return [doc_from_dict(r) for r in sorted_results]
              
        
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
        d['id'] = uuid.uuid4().hex if not doc.metadata.get('id', '') else doc.metadata['id']
        d['page_content' ] = doc.page_content
        #d['raw'] = json.dumps(d)
        if 'source' in d:
            d['source_search'] = d['source']
        if 'filepath' in d:
            d['filepath_search'] = d['filepath']
        return d





