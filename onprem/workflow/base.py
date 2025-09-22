"""
Base classes and node type system for the workflow engine.
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from langchain_core.documents import Document

from .exceptions import NodeExecutionError


# Node Type System
class NodeType:
    """Represents a node type with its connection rules."""
    def __init__(self, name: str, can_connect_to: List[str] = None, is_terminal: bool = False):
        self.name = name
        self.can_connect_to = can_connect_to or []
        self.is_terminal = is_terminal


# Define valid node type connections
NODE_TYPES = {
    "Loader": NodeType("Loader", can_connect_to=["TextSplitter", "DocumentTransformer", "Processor"]),
    "TextSplitter": NodeType("TextSplitter", can_connect_to=["TextSplitter", "DocumentTransformer", "Storage", "Processor"]),
    "DocumentTransformer": NodeType("DocumentTransformer", can_connect_to=["TextSplitter", "DocumentTransformer", "Storage", "Processor"]),
    "Storage": NodeType("Storage", is_terminal=True),
    "Query": NodeType("Query", can_connect_to=["DocumentTransformer", "Processor"]),
    "Processor": NodeType("Processor", can_connect_to=["Processor", "Exporter"]),
    "Exporter": NodeType("Exporter", is_terminal=True)
}


# Base Node Classes
class BaseNode(ABC):
    """Base class for all workflow nodes."""
    
    # Class attribute to be set by subclasses
    NODE_TYPE: str = None
    
    def __init__(self, node_id: str, config: Dict[str, Any], workflow_engine=None):
        self.node_id = node_id
        self.config = config
        self.workflow_engine = workflow_engine
        self.inputs: Dict[str, str] = {}
        self.outputs: Dict[str, str] = {}
        self._result_cache: Optional[Any] = None
    
    def get_llm(self, llm_config: Dict[str, Any]) -> Any:
        """Get LLM instance, using shared instance if available."""
        if not llm_config:
            raise NodeExecutionError(f"Node {self.node_id}: No LLM configuration provided")
        
        try:
            if self.workflow_engine:
                return self.workflow_engine.get_shared_llm(llm_config)
            else:
                # Fallback to creating new instance
                from ..llm.base import LLM
                return LLM(**llm_config)
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to initialize LLM with config {llm_config}: {str(e)}")
    
    @abstractmethod
    def get_input_types(self) -> Dict[str, str]:
        """Return dictionary mapping input port names to their expected types."""
        pass
    
    @abstractmethod
    def get_output_types(self) -> Dict[str, str]:
        """Return dictionary mapping output port names to their output types."""
        pass
    
    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the node with given inputs and return outputs."""
        pass
    
    def validate_config(self) -> bool:
        """Validate node configuration. Override in subclasses if needed."""
        return True
    
    def can_connect_to(self, target_node: 'BaseNode') -> bool:
        """Check if this node can connect to the target node based on types."""
        if not self.NODE_TYPE or not target_node.NODE_TYPE:
            return False
        
        source_type = NODE_TYPES.get(self.NODE_TYPE)
        target_type_name = target_node.NODE_TYPE
        
        if not source_type:
            return False
        
        return target_type_name in source_type.can_connect_to
    
    def is_terminal(self) -> bool:
        """Check if this node is terminal (cannot connect to other nodes)."""
        if not self.NODE_TYPE:
            return False
        
        node_type = NODE_TYPES.get(self.NODE_TYPE)
        return node_type.is_terminal if node_type else False


class LoaderNode(BaseNode):
    """Base class for document loading nodes."""
    
    NODE_TYPE = "Loader"
    
    def get_output_types(self) -> Dict[str, str]:
        return {"documents": "List[Document]"}
    
    def get_input_types(self) -> Dict[str, str]:
        return {}


class TextSplitterNode(BaseNode):
    """Base class for text splitting/chunking nodes."""
    
    NODE_TYPE = "TextSplitter"
    
    def get_input_types(self) -> Dict[str, str]:
        return {"documents": "List[Document]"}
    
    def get_output_types(self) -> Dict[str, str]:
        return {"documents": "List[Document]"}


class StorageNode(BaseNode):
    """Base class for document storage nodes."""
    
    NODE_TYPE = "Storage"
    
    def get_input_types(self) -> Dict[str, str]:
        return {"documents": "List[Document]"}
    
    def get_output_types(self) -> Dict[str, str]:
        return {"status": "str"}


class QueryNode(BaseNode):
    """Base class for querying storage indexes."""
    
    NODE_TYPE = "Query"
    
    def get_input_types(self) -> Dict[str, str]:
        return {}  # No inputs - queries existing storage
    
    def get_output_types(self) -> Dict[str, str]:
        return {"documents": "List[Document]"}
    
    def _extract_common_params(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate common query parameters."""
        import os
        
        persist_location = self.config.get("persist_location")
        query = self.config.get("query", "")
        limit = self.config.get("limit", 10)
        search_type = self.config.get("search_type")
        
        # Validate required parameters
        if not persist_location:
            raise NodeExecutionError(f"Node {self.node_id}: persist_location is required")
        if not query:
            raise NodeExecutionError(f"Node {self.node_id}: query is required")
        
        # Expand tilde (~) in persist_location to full home directory path
        persist_location = os.path.expanduser(persist_location)
        
        return {
            "persist_location": persist_location,
            "query": query,
            "limit": limit,
            "search_type": search_type
        }
    
    def _validate_search_type(self, search_type: str, valid_types: List[str]) -> None:
        """Validate search_type against supported types for this store."""
        if search_type not in valid_types:
            # Generate helpful error messages for common mismatches
            error_msg = f"Node {self.node_id}: Unknown search_type '{search_type}'. Use one of: {', '.join(valid_types)}"
            
            # Add specific suggestions for common errors
            if search_type == "hybrid" and "hybrid" not in valid_types:
                if "elasticsearch" in str(type(self)).lower():
                    error_msg += ". Note: Use ElasticsearchStore for hybrid search."
                elif "chroma" in str(type(self)).lower():
                    error_msg += ". Note: ChromaStore does not support hybrid search. Use ElasticsearchStore for hybrid search."
                elif "whoosh" in str(type(self)).lower():
                    error_msg += ". Note: WhooshStore does not support hybrid search. Use ElasticsearchStore for hybrid search."
            elif search_type == "sparse" and "sparse" not in valid_types:
                error_msg += ". Note: ChromaStore does not support sparse search. Use WhooshStore or ElasticsearchStore for sparse search."
            elif search_type == "semantic" and "semantic" not in valid_types:
                pass  # Most stores support semantic search
                
            raise NodeExecutionError(error_msg)


class ProcessorNode(BaseNode):
    """Base class for processing nodes."""
    
    NODE_TYPE = "Processor"


class DocumentProcessor(ProcessorNode):
    """Base class for processors that work with raw documents."""
    
    def get_input_types(self) -> Dict[str, str]:
        return {"documents": "List[Document]"}
    
    def get_output_types(self) -> Dict[str, str]:
        return {"results": "List[Dict]"}


class ResultProcessor(ProcessorNode):
    """Base class for processors that work with processed results."""
    
    def get_input_types(self) -> Dict[str, str]:
        return {"results": "List[Dict]"}
    
    def get_output_types(self) -> Dict[str, str]:
        return {"results": "List[Dict]"}


class AggregatorProcessor(ProcessorNode):
    """Base class for processors that aggregate multiple results into a single result."""
    
    def get_input_types(self) -> Dict[str, str]:
        return {"results": "List[Dict]"}
    
    def get_output_types(self) -> Dict[str, str]:
        return {"result": "Dict"}


class DocumentTransformerNode(BaseNode):
    """Base class for transforming documents (metadata, content, filtering, etc.)."""
    
    NODE_TYPE = "DocumentTransformer"
    
    def get_input_types(self) -> Dict[str, str]:
        return {"documents": "List[Document]"}
    
    def get_output_types(self) -> Dict[str, str]:
        return {"documents": "List[Document]"}


class ExporterNode(BaseNode):
    """Base class for exporting results to various formats."""
    
    NODE_TYPE = "Exporter"
    
    def get_input_types(self) -> Dict[str, str]:
        return {"results": "List[Dict]"}
    
    def get_output_types(self) -> Dict[str, str]:
        return {"status": "str"}