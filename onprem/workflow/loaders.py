"""
Loader node implementations for the workflow engine.
"""

import os
from typing import Dict, Any

from .. import ingest
from .base import LoaderNode
from .exceptions import NodeExecutionError


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


class LoadSpreadsheetNode(LoaderNode):
    """Loads documents from a spreadsheet using ingest.load_spreadsheet_documents."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._result_cache is not None:
            return {"documents": self._result_cache}
        
        file_path = self.config.get("file_path")
        if not file_path:
            raise NodeExecutionError(f"Node {self.node_id}: file_path is required")
        
        text_column = self.config.get("text_column")
        if not text_column:
            raise NodeExecutionError(f"Node {self.node_id}: text_column is required")
        
        metadata_columns = self.config.get("metadata_columns")
        sheet_name = self.config.get("sheet_name")
        
        try:
            documents = ingest.load_spreadsheet_documents(
                file_path=file_path,
                text_column=text_column,
                metadata_columns=metadata_columns,
                sheet_name=sheet_name
            )
            self._result_cache = documents
            return {"documents": documents}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to load spreadsheet: {str(e)}")