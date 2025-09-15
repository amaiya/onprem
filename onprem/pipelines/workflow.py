"""
YAML-based workflow execution engine for document processing pipelines.

This module allows users to define automated document processing workflows in YAML
that connect different types of nodes (Loader, TextSplitter, Storage) in a pipeline.
"""

import yaml
import os
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from langchain_core.documents import Document

from .. import ingest
from ..ingest.stores import VectorStoreFactory
from ..ingest.base import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


class WorkflowValidationError(Exception):
    """Raised when workflow validation fails."""
    pass


class NodeExecutionError(Exception):
    """Raised when node execution fails."""
    pass


# Node Type System
class NodeType:
    """Represents a node type with its connection rules."""
    def __init__(self, name: str, can_connect_to: List[str] = None, is_terminal: bool = False):
        self.name = name
        self.can_connect_to = can_connect_to or []
        self.is_terminal = is_terminal

# Define valid node type connections
NODE_TYPES = {
    "Loader": NodeType("Loader", can_connect_to=["TextSplitter"]),
    "TextSplitter": NodeType("TextSplitter", can_connect_to=["TextSplitter", "Storage"]),
    "Storage": NodeType("Storage", is_terminal=True)
    # Future extensions could easily add:
    # "Filter": NodeType("Filter", can_connect_to=["TextSplitter", "Filter", "Storage"]),
    # "Enricher": NodeType("Enricher", can_connect_to=["TextSplitter", "Storage"]),
    # "Validator": NodeType("Validator", can_connect_to=["TextSplitter", "Storage"])
}


# Base Node Classes
class BaseNode(ABC):
    """Base class for all workflow nodes."""
    
    # Class attribute to be set by subclasses
    NODE_TYPE: str = None
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        self.node_id = node_id
        self.config = config
        self.inputs: Dict[str, str] = {}
        self.outputs: Dict[str, str] = {}
        self._result_cache: Optional[Any] = None
    
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


# Concrete Loader Implementations
class LoadFromFolderNode(LoaderNode):
    """Loads documents from a folder using ingest.load_documents."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._result_cache is not None:
            return {"documents": self._result_cache}
        
        source_dir = self.config.get("source_directory")
        if not source_dir:
            raise NodeExecutionError(f"Node {self.node_id}: source_directory is required")
        
        if not os.path.exists(source_dir):
            raise NodeExecutionError(f"Node {self.node_id}: source_directory '{source_dir}' does not exist")
        
        ignored_files = self.config.get("ignored_files", [])
        kwargs = {k: v for k, v in self.config.items() 
                 if k not in ["source_directory", "ignored_files"]}
        
        try:
            documents = list(ingest.load_documents(
                source_dir, 
                ignored_files=ignored_files, 
                **kwargs
            ))
            self._result_cache = documents
            return {"documents": documents}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to load documents: {str(e)}")


class LoadSingleDocumentNode(LoaderNode):
    """Loads a single document using ingest.load_single_document."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._result_cache is not None:
            return {"documents": self._result_cache}
        
        file_path = self.config.get("file_path")
        if not file_path:
            raise NodeExecutionError(f"Node {self.node_id}: file_path is required")
        
        if not os.path.exists(file_path):
            raise NodeExecutionError(f"Node {self.node_id}: file_path '{file_path}' does not exist")
        
        kwargs = {k: v for k, v in self.config.items() if k != "file_path"}
        
        try:
            documents = ingest.load_single_document(file_path, **kwargs)
            if documents is None:
                documents = []
            self._result_cache = documents
            return {"documents": documents}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to load document: {str(e)}")


class LoadWebDocumentNode(LoaderNode):
    """Loads a document from a web URL using ingest.load_web_document."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._result_cache is not None:
            return {"documents": self._result_cache}
        
        url = self.config.get("url")
        if not url:
            raise NodeExecutionError(f"Node {self.node_id}: url is required")
        
        username = self.config.get("username")
        password = self.config.get("password")
        
        try:
            documents = ingest.load_web_document(url, username=username, password=password)
            if documents is None:
                documents = []
            self._result_cache = documents
            return {"documents": documents}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to load web document: {str(e)}")


# Concrete TextSplitter Implementations
class SplitByCharacterCountNode(TextSplitterNode):
    """Splits documents by character count using ingest.chunk_documents."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        documents = inputs.get("documents", [])
        if not documents:
            return {"documents": []}
        
        chunk_size = self.config.get("chunk_size", DEFAULT_CHUNK_SIZE)
        chunk_overlap = self.config.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
        kwargs = {k: v for k, v in self.config.items() 
                 if k not in ["chunk_size", "chunk_overlap"]}
        
        try:
            chunked_docs = ingest.chunk_documents(
                documents, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                **kwargs
            )
            return {"documents": chunked_docs}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to chunk documents: {str(e)}")


class SplitByParagraphNode(TextSplitterNode):
    """Splits documents by paragraph using ingest.chunk_documents with preserve_paragraphs=True."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        documents = inputs.get("documents", [])
        if not documents:
            return {"documents": []}
        
        chunk_size = self.config.get("chunk_size", DEFAULT_CHUNK_SIZE)
        chunk_overlap = self.config.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
        kwargs = {k: v for k, v in self.config.items() 
                 if k not in ["chunk_size", "chunk_overlap"]}
        kwargs["preserve_paragraphs"] = True
        
        try:
            chunked_docs = ingest.chunk_documents(
                documents, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                **kwargs
            )
            return {"documents": chunked_docs}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to chunk documents by paragraph: {str(e)}")


class KeepFullDocumentNode(TextSplitterNode):
    """Passes documents through without any splitting or chunking."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        documents = inputs.get("documents", [])
        return {"documents": documents}


# Concrete Storage Implementations
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


# Node Registry
NODE_REGISTRY = {
    # Loaders
    "LoadFromFolder": LoadFromFolderNode,
    "LoadSingleDocument": LoadSingleDocumentNode,
    "LoadWebDocument": LoadWebDocumentNode,
    
    # Text Splitters
    "SplitByCharacterCount": SplitByCharacterCountNode,
    "SplitByParagraph": SplitByParagraphNode,
    "KeepFullDocument": KeepFullDocumentNode,
    
    # Storage
    "ChromaStore": ChromaStoreNode,
    "WhooshStore": WhooshStoreNode,
    "ElasticsearchStore": ElasticsearchStoreNode,
}


class WorkflowEngine:
    """Executes YAML-defined workflows with validation."""
    
    def __init__(self):
        self.nodes: Dict[str, BaseNode] = {}
        self.connections: List[Dict[str, str]] = []
        self.execution_order: List[str] = []
    
    def load_workflow_from_yaml(self, yaml_path: str) -> None:
        """Load workflow definition from YAML file."""
        if not os.path.exists(yaml_path):
            raise WorkflowValidationError(f"Workflow file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r') as f:
                workflow_def = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise WorkflowValidationError(f"Invalid YAML: {str(e)}")
        
        self._parse_workflow(workflow_def)
        self._validate_workflow()
        self._determine_execution_order()
    
    def load_workflow_from_dict(self, workflow_def: Dict[str, Any]) -> None:
        """Load workflow definition from dictionary."""
        self._parse_workflow(workflow_def)
        self._validate_workflow()
        self._determine_execution_order()
    
    def _parse_workflow(self, workflow_def: Dict[str, Any]) -> None:
        """Parse workflow definition and create node instances."""
        self.nodes.clear()
        self.connections.clear()
        
        # Parse nodes
        nodes_def = workflow_def.get("nodes", {})
        for node_id, node_config in nodes_def.items():
            node_type = node_config.get("type")
            if node_type not in NODE_REGISTRY:
                raise WorkflowValidationError(f"Unknown node type: {node_type}")
            
            node_class = NODE_REGISTRY[node_type]
            node = node_class(node_id, node_config.get("config", {}))
            
            if not node.validate_config():
                raise WorkflowValidationError(f"Invalid configuration for node {node_id}")
            
            self.nodes[node_id] = node
        
        # Parse connections
        connections_def = workflow_def.get("connections", [])
        for conn in connections_def:
            if not all(key in conn for key in ["from", "from_port", "to", "to_port"]):
                raise WorkflowValidationError("Connection missing required fields: from, from_port, to, to_port")
            self.connections.append(conn)
    
    def _validate_workflow(self) -> None:
        """Validate node connections and types."""
        if not self.nodes:
            raise WorkflowValidationError("Workflow must contain at least one node")
        
        for conn in self.connections:
            source_node = self.nodes.get(conn["from"])
            target_node = self.nodes.get(conn["to"])
            
            if not source_node:
                raise WorkflowValidationError(f"Source node not found: {conn['from']}")
            if not target_node:
                raise WorkflowValidationError(f"Target node not found: {conn['to']}")
            
            # Validate port types
            source_outputs = source_node.get_output_types()
            target_inputs = target_node.get_input_types()
            
            source_port = conn["from_port"]
            target_port = conn["to_port"]
            
            if source_port not in source_outputs:
                raise WorkflowValidationError(
                    f"Source node {conn['from']} has no output port '{source_port}'. "
                    f"Available: {list(source_outputs.keys())}"
                )
            
            if target_port not in target_inputs:
                raise WorkflowValidationError(
                    f"Target node {conn['to']} has no input port '{target_port}'. "
                    f"Available: {list(target_inputs.keys())}"
                )
            
            # Validate type compatibility
            source_type = source_outputs[source_port]
            target_type = target_inputs[target_port]
            
            if source_type != target_type:
                raise WorkflowValidationError(
                    f"Type mismatch: {conn['from']}.{source_port} ({source_type}) -> "
                    f"{conn['to']}.{target_port} ({target_type})"
                )
            
            # Validate node type compatibility using the registry
            if not source_node.can_connect_to(target_node):
                source_type = source_node.NODE_TYPE or "Unknown"
                target_type = target_node.NODE_TYPE or "Unknown"
                
                # Generate helpful error message
                if source_node.is_terminal():
                    raise WorkflowValidationError(
                        f"{source_type} node '{source_node.node_id}' is terminal and cannot connect to other nodes"
                    )
                else:
                    valid_targets = NODE_TYPES.get(source_type)
                    valid_types = valid_targets.can_connect_to if valid_targets else []
                    raise WorkflowValidationError(
                        f"{source_type} node '{source_node.node_id}' cannot connect to {target_type} node '{target_node.node_id}'. "
                        f"Valid target types: {valid_types}"
                    )
    
    def _determine_execution_order(self) -> None:
        """Determine execution order using topological sort."""
        # Build dependency graph
        dependencies = {node_id: set() for node_id in self.nodes}
        dependents = {node_id: set() for node_id in self.nodes}
        
        for conn in self.connections:
            dependencies[conn["to"]].add(conn["from"])
            dependents[conn["from"]].add(conn["to"])
        
        # Topological sort
        self.execution_order = []
        no_deps = [node_id for node_id, deps in dependencies.items() if not deps]
        
        while no_deps:
            current = no_deps.pop(0)
            self.execution_order.append(current)
            
            for dependent in dependents[current]:
                dependencies[dependent].remove(current)
                if not dependencies[dependent]:
                    no_deps.append(dependent)
        
        if len(self.execution_order) != len(self.nodes):
            raise WorkflowValidationError("Workflow contains cycles")
    
    def execute(self, verbose: bool = True) -> Dict[str, Any]:
        """Execute the workflow and return results."""
        if not self.execution_order:
            raise WorkflowValidationError("Workflow not loaded or invalid")
        
        results = {}
        node_outputs = {}
        
        for node_id in self.execution_order:
            node = self.nodes[node_id]
            
            if verbose:
                print(f"Executing node: {node_id}")
            
            # Collect inputs from connected nodes
            inputs = {}
            for conn in self.connections:
                if conn["to"] == node_id:
                    source_output = node_outputs[conn["from"]]
                    inputs[conn["to_port"]] = source_output[conn["from_port"]]
            
            # Execute node
            try:
                output = node.execute(inputs)
                node_outputs[node_id] = output
                results[node_id] = output
                
                if verbose:
                    if "documents" in output:
                        print(f"  -> Processed {len(output['documents'])} documents")
                    elif "status" in output:
                        print(f"  -> {output['status']}")
            
            except Exception as e:
                error_msg = f"Failed to execute node {node_id}: {str(e)}"
                if verbose:
                    print(f"  -> ERROR: {error_msg}")
                raise NodeExecutionError(error_msg)
        
        return results


def load_workflow(yaml_path: str) -> WorkflowEngine:
    """Convenience function to load a workflow from YAML file."""
    engine = WorkflowEngine()
    engine.load_workflow_from_yaml(yaml_path)
    return engine


def execute_workflow(yaml_path: str, verbose: bool = True) -> Dict[str, Any]:
    """Convenience function to load and execute a workflow from YAML file."""
    engine = load_workflow(yaml_path)
    return engine.execute(verbose=verbose)