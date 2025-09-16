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
    "Loader": NodeType("Loader", can_connect_to=["TextSplitter", "Processor"]),  # Allow direct connection to Processor
    "TextSplitter": NodeType("TextSplitter", can_connect_to=["TextSplitter", "Storage", "Processor"]),  # Allow direct connection to Processor
    "Storage": NodeType("Storage", is_terminal=True),
    "Query": NodeType("Query", can_connect_to=["Processor"]),
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


class ProcessorNode(BaseNode):
    """Base class for processing documents (applying prompts, etc.)."""
    
    NODE_TYPE = "Processor"
    
    def get_input_types(self) -> Dict[str, str]:
        return {"documents": "List[Document]"}
    
    def get_output_types(self) -> Dict[str, str]:
        return {"results": "List[Dict]"}


class ExporterNode(BaseNode):
    """Base class for exporting results to various formats."""
    
    NODE_TYPE = "Exporter"
    
    def get_input_types(self) -> Dict[str, str]:
        return {"results": "List[Dict]"}
    
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
        
        # Use built-in pattern filtering instead of custom ignore_fn to preserve metadata
        # Extract parameters for ingest.load_documents, only excluding source_directory
        kwargs = {k: v for k, v in self.config.items() 
                 if k not in ["source_directory"]}
        
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
    """Passes documents through without any splitting or chunking.
    
    Optionally concatenates multi-page documents into single documents.
    """
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        documents = inputs.get("documents", [])
        
        concatenate_pages = self.config.get("concatenate_pages", False)
        
        if not concatenate_pages:
            return {"documents": documents}
        
        # Group documents by source file and concatenate pages
        try:
            source_groups = {}
            
            for doc in documents:
                source = doc.metadata.get('source', 'unknown')
                if source not in source_groups:
                    source_groups[source] = []
                source_groups[source].append(doc)
            
            concatenated_docs = []
            
            for source, doc_pages in source_groups.items():
                if len(doc_pages) == 1:
                    # Single page document - keep as is
                    concatenated_docs.append(doc_pages[0])
                else:
                    # Multi-page document - concatenate
                    # Sort by page number if available
                    doc_pages.sort(key=lambda x: x.metadata.get('page', 0))
                    
                    # Combine content with page breaks
                    combined_content = '\n\n--- PAGE BREAK ---\n\n'.join(
                        doc.page_content for doc in doc_pages
                    )
                    
                    # Create new document with combined metadata
                    combined_metadata = doc_pages[0].metadata.copy()
                    combined_metadata['page'] = -1  # Indicate full document
                    combined_metadata['page_count'] = len(doc_pages)
                    combined_metadata['concatenated'] = True
                    
                    # Include page range if available
                    pages = [doc.metadata.get('page', 0) for doc in doc_pages if doc.metadata.get('page', 0) > 0]
                    if pages:
                        combined_metadata['page_range'] = f"{min(pages)}-{max(pages)}"
                    
                    combined_doc = Document(
                        page_content=combined_content,
                        metadata=combined_metadata
                    )
                    concatenated_docs.append(combined_doc)
            
            return {"documents": concatenated_docs}
            
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to concatenate pages: {str(e)}")


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


# Concrete Query Implementations
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


# Concrete Processor Implementations
class PromptProcessorNode(ProcessorNode):
    """Applies a prompt to documents using an LLM."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        documents = inputs.get("documents", [])
        if not documents:
            return {"results": []}
        
        prompt_template = self.config.get("prompt", "")
        prompt_file = self.config.get("prompt_file", "")
        batch_size = self.config.get("batch_size", 5)
        
        # Get LLM configuration
        llm_config = self.config.get("llm", {"model_url": "openai://gpt-3.5-turbo"})
        
        # Load prompt from file if specified, otherwise use inline prompt
        if prompt_file:
            if not os.path.exists(prompt_file):
                raise NodeExecutionError(f"Node {self.node_id}: prompt_file '{prompt_file}' does not exist")
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt_template = f.read()
            except Exception as e:
                raise NodeExecutionError(f"Node {self.node_id}: Failed to read prompt_file '{prompt_file}': {str(e)}")
        
        if not prompt_template:
            raise NodeExecutionError(f"Node {self.node_id}: Either 'prompt' or 'prompt_file' is required")
        
        try:
            llm = self.get_llm(llm_config)
            
            results = []
            for i, doc in enumerate(documents):
                # Format the prompt with document content and metadata
                format_kwargs = {
                    'content': doc.page_content,
                    **doc.metadata  # Include all metadata
                }
                # Ensure source is available even if not in metadata
                if 'source' not in format_kwargs:
                    format_kwargs['source'] = 'Unknown'
                
                formatted_prompt = prompt_template.format(**format_kwargs)
                
                # Get LLM response
                response = llm.prompt(formatted_prompt)
                
                # Create result record
                result = {
                    'document_id': i,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'prompt': formatted_prompt,
                    'response': response,
                    'metadata': doc.metadata
                }
                results.append(result)
                
                # Progress indication
                if (i + 1) % batch_size == 0:
                    print(f"Processed {i + 1}/{len(documents)} documents")
            
            return {"results": results}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to process with prompt: {str(e)}")


class ResponseCleanerNode(ProcessorNode):
    """Cleans and post-processes LLM responses using another LLM call."""
    
    def get_input_types(self) -> Dict[str, str]:
        return {"results": "List[Dict]"}  # Takes results from PromptProcessor
    
    def get_output_types(self) -> Dict[str, str]:
        return {"results": "List[Dict]"}  # Outputs cleaned results
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = inputs.get("results", [])
        if not results:
            return {"results": []}
        
        cleanup_prompt_template = self.config.get("cleanup_prompt", "")
        cleanup_prompt_file = self.config.get("cleanup_prompt_file", "")
        
        # Get LLM configuration
        llm_config = self.config.get("llm", {"model_url": "openai://gpt-3.5-turbo"})
        
        # Load cleanup prompt from file if specified, otherwise use inline prompt
        if cleanup_prompt_file:
            if not os.path.exists(cleanup_prompt_file):
                raise NodeExecutionError(f"Node {self.node_id}: cleanup_prompt_file '{cleanup_prompt_file}' does not exist")
            try:
                with open(cleanup_prompt_file, 'r', encoding='utf-8') as f:
                    cleanup_prompt_template = f.read()
            except Exception as e:
                raise NodeExecutionError(f"Node {self.node_id}: Failed to read cleanup_prompt_file '{cleanup_prompt_file}': {str(e)}")
        
        if not cleanup_prompt_template:
            raise NodeExecutionError(f"Node {self.node_id}: Either 'cleanup_prompt' or 'cleanup_prompt_file' is required")
        
        try:
            llm = self.get_llm(llm_config)
            
            cleaned_results = []
            for i, result in enumerate(results):
                original_response = result.get('response', '')
                
                # Format cleanup prompt with the original response
                cleanup_request = cleanup_prompt_template.format(
                    original_response=original_response,
                    **result  # Include all result fields for additional context
                )
                
                # Get cleaned response
                cleaned_response = llm.prompt(cleanup_request)
                
                # Create new result with cleaned response
                cleaned_result = result.copy()
                cleaned_result['response'] = cleaned_response.strip()
                cleaned_result['original_response'] = original_response  # Keep original for reference
                cleaned_results.append(cleaned_result)
            
            return {"results": cleaned_results}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to cleanup responses: {str(e)}")


class SummaryProcessorNode(ProcessorNode):
    """Generates summaries for documents."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        documents = inputs.get("documents", [])
        if not documents:
            return {"results": []}
        
        max_length = self.config.get("max_length", 150)
        
        # Get LLM configuration
        llm_config = self.config.get("llm", {"model_url": "openai://gpt-3.5-turbo"})
        
        try:
            llm = self.get_llm(llm_config)
            
            results = []
            for i, doc in enumerate(documents):
                prompt = f"Summarize the following text in {max_length} words or less:\n\n{doc.page_content}"
                summary = llm.prompt(prompt)
                
                result = {
                    'document_id': i,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'original_length': len(doc.page_content),
                    'summary': summary,
                    'summary_length': len(summary),
                    'metadata': doc.metadata
                }
                results.append(result)
            
            return {"results": results}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to generate summaries: {str(e)}")


# Concrete Exporter Implementations
class CSVExporterNode(ExporterNode):
    """Exports results to CSV format."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = inputs.get("results", [])
        if not results:
            return {"status": "No results to export"}
        
        output_path = self.config.get("output_path", "results.csv")
        columns = self.config.get("columns", None)  # If None, use all keys from first result
        
        try:
            import csv
            
            # Determine columns
            if columns is None:
                columns = list(results[0].keys()) if results else []
            
            # Flatten nested metadata if present
            flattened_results = []
            for result in results:
                flat_result = {}
                for key, value in result.items():
                    if key == 'metadata' and isinstance(value, dict):
                        # Flatten metadata with prefix
                        for meta_key, meta_value in value.items():
                            flat_result[f"meta_{meta_key}"] = str(meta_value)
                    else:
                        flat_result[key] = str(value) if value is not None else ""
                flattened_results.append(flat_result)
            
            # Update columns to include flattened metadata
            if flattened_results:
                all_columns = set()
                for result in flattened_results:
                    all_columns.update(result.keys())
                if columns is None or 'metadata' in columns:
                    columns = sorted(list(all_columns))
            
            # Write CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                for result in flattened_results:
                    # Only include columns that exist in the result
                    row = {col: result.get(col, "") for col in columns}
                    writer.writerow(row)
            
            return {"status": f"Exported {len(results)} results to {output_path}"}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to export CSV: {str(e)}")


class ExcelExporterNode(ExporterNode):
    """Exports results to Excel format."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = inputs.get("results", [])
        if not results:
            return {"status": "No results to export"}
        
        output_path = self.config.get("output_path", "results.xlsx")
        sheet_name = self.config.get("sheet_name", "Results")
        
        try:
            import pandas as pd
            
            # Flatten results similar to CSV exporter
            flattened_results = []
            for result in results:
                flat_result = {}
                for key, value in result.items():
                    if key == 'metadata' and isinstance(value, dict):
                        for meta_key, meta_value in value.items():
                            flat_result[f"meta_{meta_key}"] = meta_value
                    else:
                        flat_result[key] = value
                flattened_results.append(flat_result)
            
            # Create DataFrame and export
            df = pd.DataFrame(flattened_results)
            df.to_excel(output_path, sheet_name=sheet_name, index=False)
            
            return {"status": f"Exported {len(results)} results to {output_path}"}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to export Excel: {str(e)}")


class JSONExporterNode(ExporterNode):
    """Exports results to JSON format."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = inputs.get("results", [])
        if not results:
            return {"status": "No results to export"}
        
        output_path = self.config.get("output_path", "results.json")
        pretty_print = self.config.get("pretty_print", True)
        
        try:
            import json
            
            with open(output_path, 'w', encoding='utf-8') as jsonfile:
                if pretty_print:
                    json.dump(results, jsonfile, indent=2, ensure_ascii=False)
                else:
                    json.dump(results, jsonfile, ensure_ascii=False)
            
            return {"status": f"Exported {len(results)} results to {output_path}"}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to export JSON: {str(e)}")


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
    
    # Query
    "QueryWhooshStore": QueryWhooshStoreNode,
    "QueryChromaStore": QueryChromaStoreNode,
    
    # Processors
    "PromptProcessor": PromptProcessorNode,
    "ResponseCleaner": ResponseCleanerNode,
    "SummaryProcessor": SummaryProcessorNode,
    
    # Exporters
    "CSVExporter": CSVExporterNode,
    "ExcelExporter": ExcelExporterNode,
    "JSONExporter": JSONExporterNode,
}


class WorkflowEngine:
    """Executes YAML-defined workflows with validation."""
    
    def __init__(self):
        self.nodes: Dict[str, BaseNode] = {}
        self.connections: List[Dict[str, str]] = []
        self.execution_order: List[str] = []
        self._shared_llm_cache: Dict[str, Any] = {}  # Cache for shared LLM instances
    
    def get_shared_llm(self, llm_config: Dict[str, Any]) -> Any:
        """Get or create a shared LLM instance based on configuration."""
        if not llm_config:
            raise WorkflowValidationError("Empty LLM configuration provided")
        
        # Create a cache key from the LLM configuration
        cache_key = str(sorted(llm_config.items()))
        
        if cache_key not in self._shared_llm_cache:
            try:
                from ..llm.base import LLM
                self._shared_llm_cache[cache_key] = LLM(**llm_config)
            except Exception as e:
                raise WorkflowValidationError(f"Failed to initialize shared LLM with config {llm_config}: {str(e)}")
        
        return self._shared_llm_cache[cache_key]
    
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
            node = node_class(node_id, node_config.get("config", {}), self)
            
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


if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path
    
    def main():
        parser = argparse.ArgumentParser(
            prog="python -m onprem.pipelines.workflow",
            description="Execute YAML-based document processing workflows",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python -m onprem.pipelines.workflow workflow.yaml
  python -m onprem.pipelines.workflow --validate workflow.yaml
  python -m onprem.pipelines.workflow --quiet workflow.yaml
  python -m onprem.pipelines.workflow --help

Workflow Structure:
  A workflow YAML file contains 'nodes' and 'connections' sections:
  
  nodes:
    my_loader:
      type: LoadFromFolder
      config:
        source_directory: "documents/"
    
    my_chunker:
      type: SplitByCharacterCount
      config:
        chunk_size: 500
        chunk_overlap: 50
  
  connections:
    - from: my_loader
      from_port: documents
      to: my_chunker
      to_port: documents

Node Types:
  Loaders:        LoadFromFolder, LoadSingleDocument, LoadWebDocument
  TextSplitters:  SplitByCharacterCount, SplitByParagraph, KeepFullDocument
  Storage:        ChromaStore, WhooshStore, ElasticsearchStore
  Query:          QueryWhooshStore, QueryChromaStore
  Processors:     PromptProcessor, ResponseCleaner, SummaryProcessor
  Exporters:      CSVExporter, ExcelExporter, JSONExporter

For detailed documentation, see: nbs/tests/workflows/workflow_tutorial.md
            """
        )
        
        parser.add_argument(
            "workflow_file",
            nargs="?",
            help="Path to the YAML workflow file to execute"
        )
        
        parser.add_argument(
            "-v", "--verbose", 
            action="store_true",
            default=True,
            help="Show detailed progress information (default: enabled)"
        )
        
        parser.add_argument(
            "-q", "--quiet",
            action="store_true", 
            help="Suppress progress output"
        )
        
        parser.add_argument(
            "--validate",
            action="store_true",
            help="Validate workflow without executing it"
        )
        
        parser.add_argument(
            "--list-nodes",
            action="store_true",
            help="List all available node types"
        )
        
        parser.add_argument(
            "--version",
            action="version",
            version="OnPrem Workflow Engine v1.0"
        )
        
        args = parser.parse_args()
        
        # Handle special commands that don't require workflow file
        if args.list_nodes:
            print("📋 Available Node Types:\n")
            
            categories = {
                "Loader Nodes": ["LoadFromFolder", "LoadSingleDocument", "LoadWebDocument"],
                "TextSplitter Nodes": ["SplitByCharacterCount", "SplitByParagraph", "KeepFullDocument"], 
                "Storage Nodes": ["ChromaStore", "WhooshStore", "ElasticsearchStore"],
                "Query Nodes": ["QueryWhooshStore", "QueryChromaStore"],
                "Processor Nodes": ["PromptProcessor", "ResponseCleaner", "SummaryProcessor"],
                "Exporter Nodes": ["CSVExporter", "ExcelExporter", "JSONExporter"]
            }
            
            for category, nodes in categories.items():
                print(f"  {category}:")
                for node in nodes:
                    print(f"    - {node}")
                print()
            
            return 0
        
        # For other commands, workflow file is required
        if not args.workflow_file:
            parser.error("workflow_file is required unless using --list-nodes")
        
        # Check if workflow file exists
        workflow_path = Path(args.workflow_file)
        if not workflow_path.exists():
            print(f"❌ Workflow file not found: {args.workflow_file}")
            return 1
        
        # Determine verbosity
        verbose = args.verbose and not args.quiet
        
        # Validate or execute workflow
        try:
            if args.validate:
                print(f"🔍 Validating workflow: {args.workflow_file}")
                engine = load_workflow(str(workflow_path))
                print("✅ Workflow validation successful!")
                
                # Show workflow summary
                if verbose:
                    node_count = len(engine.nodes)
                    conn_count = len(engine.connections)
                    print(f"📊 Workflow summary: {node_count} nodes, {conn_count} connections")
                    
                    # Group nodes by type
                    node_types = {}
                    for node in engine.nodes.values():
                        node_type = node.NODE_TYPE
                        if node_type not in node_types:
                            node_types[node_type] = 0
                        node_types[node_type] += 1
                    
                    print("📋 Node breakdown:")
                    for node_type, count in sorted(node_types.items()):
                        print(f"  - {node_type}: {count}")
                
                return 0
            else:
                print(f"🚀 Executing workflow: {args.workflow_file}")
                results = execute_workflow(str(workflow_path), verbose=verbose)
                print(f"\n✅ Workflow completed successfully!")
                
                if verbose:
                    # Show execution summary
                    total_docs = 0
                    exports = []
                    
                    for node_id, result in results.items():
                        if isinstance(result, dict):
                            if "documents" in result and isinstance(result["documents"], list):
                                total_docs += len(result["documents"])
                            elif "status" in result and ("exported" in result["status"].lower() or "saved" in result["status"].lower()):
                                exports.append(result["status"])
                    
                    if total_docs > 0:
                        print(f"📄 Total documents processed: {total_docs}")
                    
                    if exports:
                        print("💾 Export results:")
                        for export in exports:
                            print(f"  - {export}")
                
                return 0
                
        except WorkflowValidationError as e:
            print(f"❌ Workflow validation failed: {e}")
            return 1
        except NodeExecutionError as e:
            print(f"❌ Workflow execution failed: {e}")
            return 1
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    sys.exit(main())