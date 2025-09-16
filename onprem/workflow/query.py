"""Query node implementations for the workflow engine."""

from typing import Dict, Any

from ..ingest.stores import VectorStoreFactory
from .base import QueryNode
from .exceptions import NodeExecutionError


class QueryWhooshStoreNode(QueryNode):
    """Queries documents from a Whoosh search index."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        persist_location = self.config.get("persist_location")
        query = self.config.get("query", "")
        limit = self.config.get("limit", 100)
        
        if not persist_location:
            raise NodeExecutionError(f"Node {self.node_id}: persist_location is required")
        if not query:
            raise NodeExecutionError(f"Node {self.node_id}: query is required")
        
        try:
            store = VectorStoreFactory.create("whoosh", persist_location=persist_location)
            # Use the search method with return_dict=False to get Document objects
            search_results = store.search(query, limit=limit, return_dict=False)
            
            # Extract documents from the results structure
            documents = search_results.get('hits', []) if isinstance(search_results, dict) else search_results
            return {"documents": documents}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to query Whoosh: {str(e)}")


class QueryChromaStoreNode(QueryNode):
    """Queries documents from a ChromaDB vector store."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        persist_location = self.config.get("persist_location")
        query = self.config.get("query", "")
        limit = self.config.get("limit", 10)
        
        if not persist_location:
            raise NodeExecutionError(f"Node {self.node_id}: persist_location is required")
        if not query:
            raise NodeExecutionError(f"Node {self.node_id}: query is required")
        
        try:
            store = VectorStoreFactory.create("chroma", persist_location=persist_location)
            # Use the search method (similar to WhooshStore)
            results = store.search(query, limit=limit, return_dict=False)
            return {"documents": results}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to query Chroma: {str(e)}")