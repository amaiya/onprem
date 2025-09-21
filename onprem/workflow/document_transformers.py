"""
Document transformer node implementations for workflow pipelines.

This module contains nodes that transform documents while preserving the
List[Document] -> List[Document] flow, including metadata enrichment,
content modification, filtering, and custom transformations.
"""

import os
import copy
from typing import Dict, List, Any
from langchain_core.documents import Document

from .base import DocumentTransformerNode
from .exceptions import NodeExecutionError


class AddMetadataNode(DocumentTransformerNode):
    """Adds static metadata fields to all documents."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        documents = inputs.get("documents", [])
        if not documents:
            return {"documents": []}
        
        # Get metadata to add from config
        metadata_to_add = self.config.get("metadata", {})
        if not metadata_to_add:
            raise NodeExecutionError(f"Node {self.node_id}: 'metadata' config is required")
        
        # Validate metadata is a dictionary
        if not isinstance(metadata_to_add, dict):
            raise NodeExecutionError(f"Node {self.node_id}: 'metadata' must be a dictionary")
        
        try:
            enriched_documents = []
            for doc in documents:
                # Create a copy to avoid modifying original
                new_doc = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, **metadata_to_add}  # Merge metadata
                )
                enriched_documents.append(new_doc)
            
            return {"documents": enriched_documents}
            
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to add metadata: {str(e)}")


class ContentPrefixNode(DocumentTransformerNode):
    """Prepends text to the page_content of all documents."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        documents = inputs.get("documents", [])
        if not documents:
            return {"documents": []}
        
        prefix = self.config.get("prefix", "")
        if not prefix:
            raise NodeExecutionError(f"Node {self.node_id}: 'prefix' config is required")
        
        # Optional separator between prefix and content
        separator = self.config.get("separator", "\n\n")
        
        try:
            modified_documents = []
            for doc in documents:
                new_content = prefix + separator + doc.page_content
                new_doc = Document(
                    page_content=new_content,
                    metadata=doc.metadata.copy()
                )
                modified_documents.append(new_doc)
            
            return {"documents": modified_documents}
            
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to add prefix: {str(e)}")


class ContentSuffixNode(DocumentTransformerNode):
    """Appends text to the page_content of all documents."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        documents = inputs.get("documents", [])
        if not documents:
            return {"documents": []}
        
        suffix = self.config.get("suffix", "")
        if not suffix:
            raise NodeExecutionError(f"Node {self.node_id}: 'suffix' config is required")
        
        # Optional separator between content and suffix
        separator = self.config.get("separator", "\n\n")
        
        try:
            modified_documents = []
            for doc in documents:
                new_content = doc.page_content + separator + suffix
                new_doc = Document(
                    page_content=new_content,
                    metadata=doc.metadata.copy()
                )
                modified_documents.append(new_doc)
            
            return {"documents": modified_documents}
            
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to add suffix: {str(e)}")


class DocumentFilterNode(DocumentTransformerNode):
    """Filters documents based on metadata or content criteria."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        documents = inputs.get("documents", [])
        if not documents:
            return {"documents": []}
        
        # Get filter criteria from config
        metadata_filters = self.config.get("metadata_filters", {})
        content_contains = self.config.get("content_contains", [])
        content_excludes = self.config.get("content_excludes", [])
        min_length = self.config.get("min_length", 0)
        max_length = self.config.get("max_length", float('inf'))
        
        try:
            filtered_documents = []
            for doc in documents:
                # Check metadata filters
                passes_metadata = True
                for key, value in metadata_filters.items():
                    doc_value = doc.metadata.get(key)
                    if doc_value != value:
                        passes_metadata = False
                        break
                
                if not passes_metadata:
                    continue
                
                # Check content length
                content_length = len(doc.page_content)
                if content_length < min_length or content_length > max_length:
                    continue
                
                # Check content contains
                if content_contains:
                    passes_contains = any(term.lower() in doc.page_content.lower() 
                                        for term in content_contains)
                    if not passes_contains:
                        continue
                
                # Check content excludes
                if content_excludes:
                    passes_excludes = not any(term.lower() in doc.page_content.lower() 
                                            for term in content_excludes)
                    if not passes_excludes:
                        continue
                
                # Document passes all filters
                filtered_documents.append(doc)
            
            return {"documents": filtered_documents}
            
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to filter documents: {str(e)}")


class PythonDocumentTransformerNode(DocumentTransformerNode):
    """Executes custom Python code to transform documents with security controls."""
    
    def _load_code(self, node_id: str, config: Dict[str, Any]) -> str:
        """Load Python code from config or file."""
        python_code = config.get("code", "")
        code_file = config.get("code_file", "")
        
        # Load code from file if specified
        if code_file:
            if not os.path.exists(code_file):
                raise NodeExecutionError(f"Node {node_id}: code_file '{code_file}' does not exist")
            try:
                with open(code_file, 'r', encoding='utf-8') as f:
                    python_code = f.read()
            except Exception as e:
                raise NodeExecutionError(f"Node {node_id}: Failed to read code_file '{code_file}': {str(e)}")
        
        if not python_code:
            raise NodeExecutionError(f"Node {node_id}: Either 'code' or 'code_file' is required")
            
        return python_code
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create a safe execution environment with restricted builtins."""
        return {
            '__builtins__': {
                # Basic operations
                'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
                'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                'min': min, 'max': max, 'sum': sum, 'abs': abs,
                'round': round, 'sorted': sorted, 'reversed': reversed,
                'enumerate': enumerate, 'zip': zip, 'range': range,
                # String and iteration
                'print': print,  # For debugging
                'any': any, 'all': all,
            },
            # Safe modules (pre-imported, no __import__ needed)
            're': __import__('re'),
            'json': __import__('json'), 
            'math': __import__('math'),
            'datetime': __import__('datetime'),
            # Provide Document class for creating new documents
            'Document': Document,
        }
    
    def _execute_code_safely(self, node_id: str, code: str, local_vars: Dict[str, Any], item_id: int) -> Dict[str, Any]:
        """Execute Python code safely and return the result."""
        safe_globals = self._create_safe_globals()
        
        try:
            exec(code, safe_globals, local_vars)
            return local_vars
        except Exception as e:
            raise NodeExecutionError(f"Node {node_id}: Error executing Python code for document {item_id}: {str(e)}")
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        documents = inputs.get("documents", [])
        if not documents:
            return {"documents": []}
        
        # Load Python code
        python_code = self._load_code(self.node_id, self.config)
        
        try:
            transformed_documents = []
            for i, doc in enumerate(documents):
                # Set up local variables for this document
                local_vars = {
                    'doc': doc,
                    'document': doc,  # Alias
                    'content': doc.page_content,
                    'metadata': doc.metadata.copy(),  # Mutable copy
                    'document_id': i,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'transformed_doc': None  # For the user to set
                }
                
                # Execute the user's Python code
                executed_vars = self._execute_code_safely(
                    self.node_id, python_code, local_vars, i
                )
                
                # Get the transformed document
                transformed_doc = executed_vars.get('transformed_doc')
                
                if transformed_doc is None:
                    # If user didn't set transformed_doc, create one from modified variables
                    new_content = executed_vars.get('content', doc.page_content)
                    new_metadata = executed_vars.get('metadata', doc.metadata)
                    transformed_doc = Document(
                        page_content=new_content,
                        metadata=new_metadata
                    )
                elif not isinstance(transformed_doc, Document):
                    raise NodeExecutionError(f"Node {self.node_id}: transformed_doc must be a Document object, got {type(transformed_doc)}")
                
                transformed_documents.append(transformed_doc)
            
            return {"documents": transformed_documents}
            
        except Exception as e:
            if "Error executing Python code" in str(e):
                raise  # Re-raise execution errors as-is
            raise NodeExecutionError(f"Node {self.node_id}: Failed to transform documents: {str(e)}")


