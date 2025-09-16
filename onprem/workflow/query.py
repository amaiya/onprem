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


class QueryElasticsearchStoreNode(QueryNode):
    """Queries documents from an Elasticsearch index."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        persist_location = self.config.get("persist_location")
        query = self.config.get("query", "")
        limit = self.config.get("limit", 10)
        search_type = self.config.get("search_type", "sparse")  # sparse, semantic, or hybrid
        
        if not persist_location:
            raise NodeExecutionError(f"Node {self.node_id}: persist_location is required")
        if not query:
            raise NodeExecutionError(f"Node {self.node_id}: query is required")
        
        try:
            # Extract additional Elasticsearch-specific parameters
            index_name = self.config.get("index_name", "default_index")
            
            store_params = {
                "kind": "elasticsearch",
                "persist_location": persist_location,
                "index_name": index_name
            }
            
            # Add optional authentication parameters
            basic_auth = self.config.get("basic_auth")
            if basic_auth:
                store_params["basic_auth"] = basic_auth
            
            # Add optional SSL parameters
            verify_certs = self.config.get("verify_certs")
            if verify_certs is not None:
                store_params["verify_certs"] = verify_certs
            
            ca_certs = self.config.get("ca_certs")
            if ca_certs:
                store_params["ca_certs"] = ca_certs
                
            timeout = self.config.get("timeout")
            if timeout:
                store_params["timeout"] = timeout
            
            store = VectorStoreFactory.create(**store_params)
            
            # Perform search based on search_type
            if search_type == "sparse":
                # Standard text search
                results = store.search(query, limit=limit, return_dict=False)
                return {"documents": results}
            elif search_type == "semantic":
                # Semantic/dense search
                results = store.semantic_search(query, limit=limit, return_dict=False)
                return {"documents": results}
            elif search_type == "hybrid":
                # Hybrid search (if available)
                if hasattr(store, 'hybrid_search'):
                    weights = self.config.get("weights", [0.6, 0.4])  # Default weights
                    results = store.hybrid_search(query, limit=limit, weights=weights)
                    return {"documents": results}
                else:
                    # Fallback to semantic search
                    results = store.semantic_search(query, limit=limit, return_dict=False)
                    return {"documents": results}
            else:
                raise NodeExecutionError(f"Node {self.node_id}: Unknown search_type '{search_type}'. Use 'sparse', 'semantic', or 'hybrid'")
                
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to query Elasticsearch: {str(e)}")