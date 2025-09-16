"""
Processor node implementations for workflow pipelines.

This module contains processor nodes that apply various types of processing
to documents or results using LLMs or other processing methods.
"""

import os
from typing import Dict, List, Any
from langchain_core.documents import Document

from .base import DocumentProcessor, ResultProcessor
from .exceptions import NodeExecutionError


class PromptProcessorNode(DocumentProcessor):
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


class ResponseCleanerNode(ResultProcessor):
    """Cleans and post-processes LLM responses using another LLM call."""
    
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


class SummaryProcessorNode(DocumentProcessor):
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