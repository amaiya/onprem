"""Storage node implementations for the workflow engine."""

from typing import Dict, Any

from ..ingest.stores import VectorStoreFactory
from .base import StorageNode
from .exceptions import NodeExecutionError


class ChromaStoreNode(StorageNode):
    """Stores documents in a ChromaDB vector store."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        documents = inputs.get("documents", [])
        if not documents:
            return {"status": "No documents to store"}
        
        persist_location = self.config.get("persist_location")
        kwargs = {k: v for k, v in self.config.items() if k != "persist_location"}
        
        try:
            store = VectorStoreFactory.create("chroma", persist_location=persist_location, **kwargs)
            store.add_documents(documents)
            return {"status": f"Successfully stored {len(documents)} documents in ChromaStore"}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to store in Chroma: {str(e)}")


class WhooshStoreNode(StorageNode):
    """Stores documents in a Whoosh search index."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        documents = inputs.get("documents", [])
        if not documents:
            return {"status": "No documents to store"}
        
        persist_location = self.config.get("persist_location")
        kwargs = {k: v for k, v in self.config.items() if k != "persist_location"}
        
        try:
            store = VectorStoreFactory.create("whoosh", persist_location=persist_location, **kwargs)
            store.add_documents(documents)
            return {"status": f"Successfully stored {len(documents)} documents in WhooshStore"}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to store in Whoosh: {str(e)}")


class ElasticsearchStoreNode(StorageNode):
    """Stores documents in an Elasticsearch index."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        documents = inputs.get("documents", [])
        if not documents:
            return {"status": "No documents to store"}
        
        persist_location = self.config.get("persist_location")
        kwargs = {k: v for k, v in self.config.items() if k != "persist_location"}
        
        try:
            store = VectorStoreFactory.create("elasticsearch", persist_location=persist_location, **kwargs)
            store.add_documents(documents)
            return {"status": f"Successfully stored {len(documents)} documents in ElasticsearchStore"}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to store in Elasticsearch: {str(e)}")