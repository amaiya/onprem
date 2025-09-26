"""
Text splitter node implementations for the workflow engine.
"""

from typing import Dict, Any, List
from langchain_core.documents import Document

from .. import ingest
from ..ingest.base import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, _concatenate_document_pages, _truncate_documents
from .base import TextSplitterNode
from .exceptions import NodeExecutionError


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
    
    Optionally concatenates multi-page documents into single documents
    and/or truncates documents to a maximum number of words.
    
    Configuration options:
    - concatenate_pages: bool - Combine multi-page documents into single documents
    - max_words: int - Truncate document content to first N words (applied after concatenation)
    """
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        documents = inputs.get("documents", [])
        
        concatenate_pages = self.config.get("concatenate_pages", False)
        max_words = self.config.get("max_words", None)
        
        # First handle concatenation if requested
        if concatenate_pages:
            documents = _concatenate_document_pages(documents)
        
        # Then handle word truncation if requested
        if max_words and max_words > 0:
            documents = _truncate_documents(documents, max_words)
        
        return {"documents": documents}