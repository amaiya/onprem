"""Query node implementations for the workflow engine."""

import os
from typing import Dict, Any

from ..ingest.stores import VectorStoreFactory
from .base import QueryNode
from .exceptions import NodeExecutionError


class QueryWhooshStoreNode(QueryNode):
    """Queries documents from a Whoosh search index."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Extract common parameters
        params = self._extract_common_params(inputs)
        persist_location = params["persist_location"]
        query = params["query"]
        limit = params.get("limit", 100)  # Default limit for Whoosh
        search_type = params.get("search_type") if params.get("search_type") is not None else "sparse"  # Default for Whoosh
        
        # Validate search_type - WhooshStore supports sparse and semantic only
        self._validate_search_type(search_type, ["sparse", "semantic"])
        
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
        # Extract common parameters
        params = self._extract_common_params(inputs)
        persist_location = params["persist_location"]
        query = params["query"]
        limit = params.get("limit", 10)  # Default limit for Chroma
        search_type = params.get("search_type") if params.get("search_type") is not None else "semantic"  # Default for Chroma
        
        # Validate search_type - ChromaStore supports semantic only
        self._validate_search_type(search_type, ["semantic"])
        
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
        # Extract common parameters
        params = self._extract_common_params(inputs)
        persist_location = params["persist_location"]
        query = params["query"]
        limit = params.get("limit", 10)  # Default limit for Elasticsearch
        search_type = params.get("search_type") if params.get("search_type") is not None else "sparse"  # Default for Elasticsearch
        
        # Validate search_type - Elasticsearch supports all types
        self._validate_search_type(search_type, ["sparse", "semantic", "hybrid"])
        
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
        # Extract common parameters
        params = self._extract_common_params(inputs)
        persist_location = params["persist_location"]
        query = params["query"]
        limit = params.get("limit", 10)  # Default limit for DualStore
        search_type = params.get("search_type") if params.get("search_type") is not None else "hybrid"  # Default for DualStore
        
        # Validate search_type - DualStore supports all types
        self._validate_search_type(search_type, ["sparse", "semantic", "hybrid"])
        
        try:
            # Provide default embedding parameters for DualStore initialization
            store = VectorStoreFactory.create(
                "chroma+whoosh", 
                persist_location=persist_location,
                embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                embedding_model_kwargs={},
                embedding_encode_kwargs={"normalize_embeddings": False}
            )
            
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
