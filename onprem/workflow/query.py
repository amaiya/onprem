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
        search_type = self.config.get("search_type", "sparse")  # sparse or semantic
        
        if not persist_location:
            raise NodeExecutionError(f"Node {self.node_id}: persist_location is required")
        if not query:
            raise NodeExecutionError(f"Node {self.node_id}: query is required")
        
        # Validate search_type - WhooshStore supports sparse and semantic only
        valid_search_types = ["sparse", "semantic"]
        if search_type not in valid_search_types:
            if search_type == "hybrid":
                raise NodeExecutionError(f"Node {self.node_id}: WhooshStore does not support hybrid search. Use ElasticsearchStore for hybrid search.")
            else:
                raise NodeExecutionError(f"Node {self.node_id}: Unknown search_type '{search_type}'. WhooshStore supports 'sparse' or 'semantic'")
        
        try:
            store = VectorStoreFactory.create("whoosh", persist_location=persist_location)
            
            # Perform search based on search_type
            if search_type == "sparse":
                # Pure keyword/full-text search
                search_results = store.search(query, limit=limit, return_dict=False)
                documents = search_results.get('hits', []) if isinstance(search_results, dict) else search_results
                return {"documents": documents}
            elif search_type == "semantic":
                # Semantic search (keyword + embedding re-ranking)
                results = store.semantic_search(query, limit=limit)
                return {"documents": results}
                
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to query Whoosh: {str(e)}")


class QueryChromaStoreNode(QueryNode):
    """Queries documents from a ChromaDB vector store."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        persist_location = self.config.get("persist_location")
        query = self.config.get("query", "")
        limit = self.config.get("limit", 10)
        search_type = self.config.get("search_type", "semantic")  # semantic only
        
        if not persist_location:
            raise NodeExecutionError(f"Node {self.node_id}: persist_location is required")
        if not query:
            raise NodeExecutionError(f"Node {self.node_id}: query is required")
        
        # Validate search_type - ChromaStore supports semantic only
        if search_type != "semantic":
            if search_type == "sparse":
                raise NodeExecutionError(f"Node {self.node_id}: ChromaStore does not support sparse search. Use WhooshStore or ElasticsearchStore for sparse search.")
            elif search_type == "hybrid":
                raise NodeExecutionError(f"Node {self.node_id}: ChromaStore does not support hybrid search. Use ElasticsearchStore for hybrid search.")
            else:
                raise NodeExecutionError(f"Node {self.node_id}: Unknown search_type '{search_type}'. ChromaStore supports 'semantic' only")
        
        try:
            store = VectorStoreFactory.create("chroma", persist_location=persist_location)
            # ChromaStore: search() == semantic_search()
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
        
        # Validate search_type early, before connecting to Elasticsearch
        valid_search_types = ["sparse", "semantic", "hybrid"]
        if search_type not in valid_search_types:
            raise NodeExecutionError(f"Node {self.node_id}: Unknown search_type '{search_type}'. Use 'sparse', 'semantic', or 'hybrid'")
        
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
                # Standard text search - use exact same approach as WhooshStore
                search_results = store.search(query, limit=limit, return_dict=False)
                # Extract documents from the results structure (consistent with WhooshStore)
                documents = search_results.get('hits', []) if isinstance(search_results, dict) else search_results
                return {"documents": documents}
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
                
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to query Elasticsearch: {str(e)}")


class QueryDualStoreNode(QueryNode):
    """Queries documents from a dual vector store (combining sparse and dense search)."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        persist_location = self.config.get("persist_location")
        query = self.config.get("query", "")
        limit = self.config.get("limit", 10)
        search_type = self.config.get("search_type", "hybrid")  # sparse, semantic, or hybrid
        
        if not persist_location:
            raise NodeExecutionError(f"Node {self.node_id}: persist_location is required")
        if not query:
            raise NodeExecutionError(f"Node {self.node_id}: query is required")
        
        # Validate search_type
        valid_search_types = ["sparse", "semantic", "hybrid"]
        if search_type not in valid_search_types:
            raise NodeExecutionError(f"Node {self.node_id}: Unknown search_type '{search_type}'. Use 'sparse', 'semantic', or 'hybrid'")
        
        try:
            store = VectorStoreFactory.create("dual", persist_location=persist_location)
            
            # Perform search based on search_type
            if search_type == "sparse":
                # Use sparse search from the dual store
                results = store.search(query, limit=limit, return_dict=False)
                # Handle both dict and list return formats
                documents = results.get('hits', []) if isinstance(results, dict) else results
                return {"documents": documents}
            elif search_type == "semantic":
                # Use semantic search from the dual store
                results = store.semantic_search(query, limit=limit)
                return {"documents": results}
            elif search_type == "hybrid":
                # Use hybrid search combining both sparse and dense
                weights = self.config.get("weights", [0.6, 0.4])  # [dense_weight, sparse_weight]
                results = store.hybrid_search(query, limit=limit, weights=weights)
                return {"documents": results}
                
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to query dual store: {str(e)}")