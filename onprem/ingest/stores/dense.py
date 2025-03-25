"""vector database for question-answering and other tasks"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/01_ingest.stores.dense.ipynb.

# %% auto 0
__all__ = ['COLLECTION_NAME', 'DenseStore']

# %% ../../../nbs/01_ingest.stores.dense.ipynb 3
import os
import os.path
from typing import List, Optional, Callable, Dict, Sequence
from tqdm import tqdm

from ..helpers import doc_from_dict
from ...utils import get_datadir, DEFAULT_DB
from ..base import batchify_chunks, process_folder, does_vectorstore_exist, VectorStore
from ..base import CHROMA_MAX
try:
    from langchain_chroma import Chroma
    import chromadb
    from chromadb.config import Settings
    CHROMA_INSTALLED = True
except ImportError:
    CHROMA_INSTALLED = False


os.environ["TOKENIZERS_PARALLELISM"] = "0"
COLLECTION_NAME = "onprem_chroma"


class DenseStore(VectorStore):
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        **kwargs
    ):
        """
        Ingests all documents in `source_folder` (previously-ingested documents are ignored)

        **Args**:

          - *persist_directory*: Path to vector database (created if it doesn't exist).
                                 Default is `onprem_data/vectordb` in user's home directory.
          - *embedding_model*: name of sentence-transformers model
          - *embedding_model_kwargs*: arguments to embedding model (e.g., `{device':'cpu'}`). If None, GPU used if available.
          - *embedding_encode_kwargs*: arguments to encode method of
                                       embedding model (e.g., `{'normalize_embeddings': False}`).


        **Returns**: `None`
        """
        if not CHROMA_INSTALLED:
            raise ImportError('Please install chromadb: pip install chromadb langchain_chroma')

        from langchain_chroma import Chroma
        import chromadb
        from chromadb.config import Settings

        self.persist_directory = persist_directory or os.path.join(
            get_datadir(), DEFAULT_DB
        )
        self.init_embedding_model(**kwargs) # stored in self.embeddings

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


    def exists(self):
        return self.get_db() is not None


    def add_documents(self, documents, batch_size:int=CHROMA_MAX):
        """
        Stores instances of `langchain_core.documents.base.Document` in vectordb
        """
        if not documents:
            return
        db = self.get_db()
        if db:
            print("Creating embeddings. May take some minutes...")
            chunk_batches, total_chunks = batchify_chunks(documents, batch_size=batch_size)
            for lst in tqdm(chunk_batches, total=total_chunks):
                db.add_documents(lst)
        else:
            chunk_batches, total_chunks = batchify_chunks(documents, batch_size)
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


    def remove_document(self, id_to_delete):
        """
        Remove a single document with ID, `id_to_delete`.
        """
        if not self.exists(): return
        self.get_db().delete(ids=[id_to_delete])
        return

    def update_documents(self,
                         doc_dicts:dict, # dictionary with keys 'page_content', 'source', 'id', etc.
                         **kwargs):

        """
        Update a set of documents (doc in index with same ID will be over-written)
        """
        self.check()
        db = self.get_db()
        docs = [doc_from_dict(d) for d in doc_dicts]
        ids = [d['id'] for d in doc_dicts]
        return db.update_documents(ids, docs)


    def _convert_to_dict(self, raw_results):
        """
        Convert raw results to dictionary
        """
        ids = raw_results['ids']
        texts = raw_results['documents']
        metadatas = raw_results['metadatas']
        results = []
        for i, m in enumerate(metadatas):
            m['page_content'] = texts[i]
            m['id'] = ids[i]
            results.append(m)
        return results

    
    def get_all_docs(self):
        """
        Returns all docs
        """
        if not self.exists(): return []

        raw_results =  self.get_db().get()
        return self._convert_to_dict(raw_results)


    def get_doc(self, id):
        """
        Retrieve a record by ID
        """
        if not self.exists(): return None
        raw_results = self.get_db().get(ids=[id])
        return self._convert_to_dict(raw_results)[0] if len(raw_results['ids']) > 0 else None

    
    def get_size(self):
        """
        Get total number of records
        """
        if not self.exists(): return 0
        return len(self.get_db().get()['documents'])

    
    def erase(self, confirm=True):
        """
        Resets collection and removes and stored documents
        """
        if not self.exists(): return True
        shall = True
        if confirm:
            msg = (
                f"You are about to remove all documents from the vector database."
                + f"(Original documents on file system will remain.) Are you sure?"
            )
            shall = input("%s (Y/n) " % msg) == "Y"
        if shall:
            self.get_db().reset_collection()
            return True
        return False


    def query(self,
              query:str, # query string
              k:int = 4, # max number of results to return
              filters:Optional[Dict[str, str]] = None, # filter sources by metadata values using Chroma metadata syntax (e.g., {'table':True})
              where_document:Optional[Dict[str, str]] = None, # filter sources by document content in Chroma syntax (e.g., {"$contains": "Canada"})
              **kwargs):
        """
        Perform a semantic search of the vector DB
        """
        if not self.exists(): return []
        db = self.get_db()
        results = db.similarity_search_with_score(query, 
                                                  filter=filters,
                                                  where_document=where_document,
                                                  k = k, **kwargs)
        if not results: return []
        docs, scores = zip(*results)
        for doc, score in zip(docs, scores):
            simscore = 1 - score
            doc.metadata["score"] = 1-score
        return docs      

    def semantic_search(self, *args, **kwargs):
        """
        Semantic search is equivalent to queries in this class
        """
        return self.query(*args, **kwargs)

    

