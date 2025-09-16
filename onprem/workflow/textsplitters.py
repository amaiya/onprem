"""
Text splitter node implementations for the workflow engine.
"""

from typing import Dict, Any, List
from langchain_core.documents import Document

from .. import ingest
from ..ingest.base import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
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
            documents = self._concatenate_pages(documents)
        
        # Then handle word truncation if requested
        if max_words and max_words > 0:
            documents = self._truncate_documents(documents, max_words)
        
        return {"documents": documents}
    
    def _truncate_documents(self, documents: List[Any], max_words: int) -> List[Any]:
        """Truncate documents to maximum number of words."""
        truncated_docs = []
        
        for doc in documents:
            content = doc.page_content
            words = content.split()
            
            if len(words) > max_words:
                # Truncate to max_words
                truncated_content = ' '.join(words[:max_words])
                
                # Create new document with truncated content
                new_metadata = doc.metadata.copy()
                new_metadata['original_word_count'] = len(words)
                new_metadata['truncated'] = True
                new_metadata['truncated_word_count'] = max_words
                
                from langchain_core.documents import Document
                truncated_doc = Document(
                    page_content=truncated_content,
                    metadata=new_metadata
                )
                truncated_docs.append(truncated_doc)
            else:
                # Document is already under the limit
                truncated_docs.append(doc)
        
        return truncated_docs
    
    def _concatenate_pages(self, documents: List[Any]) -> List[Any]:
        """Concatenate multi-page documents by source file."""
        if not documents:
            return documents
        
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
                    
                    from langchain_core.documents import Document
                    combined_doc = Document(
                        page_content=combined_content,
                        metadata=combined_metadata
                    )
                    concatenated_docs.append(combined_doc)
            
            return concatenated_docs
            
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to concatenate pages: {str(e)}")