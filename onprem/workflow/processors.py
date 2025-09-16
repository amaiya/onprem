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


class PythonCodeMixin:
    """Shared functionality for executing Python code safely in workflow nodes."""
    
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
    
    def _execute_code_safely(self, node_id: str, code: str, local_vars: Dict[str, Any], item_id: int, item_type: str = "item") -> Dict[str, Any]:
        """Execute Python code safely and return the result."""
        safe_globals = self._create_safe_globals()
        
        try:
            exec(code, safe_globals, local_vars)
            return local_vars
        except Exception as e:
            raise NodeExecutionError(f"Node {node_id}: Error executing Python code for {item_type} {item_id}: {str(e)}")


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


class PythonDocumentProcessorNode(DocumentProcessor, PythonCodeMixin):
    """Executes custom Python code on documents with proper security controls."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        documents = inputs.get("documents", [])
        if not documents:
            return {"results": []}
        
        # Load Python code
        python_code = self._load_code(self.node_id, self.config)
        
        try:
            results = []
            for i, doc in enumerate(documents):
                # Set up local variables for this document
                local_vars = {
                    'doc': doc,
                    'document': doc,  # Alias
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'document_id': i,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'result': {}  # For the user to populate
                }
                
                # Execute the user's Python code
                executed_vars = self._execute_code_safely(
                    self.node_id, python_code, local_vars, i, "document"
                )
                
                # Get the result from executed variables
                result = executed_vars.get('result', {})
                
                # Ensure result is a dictionary and add metadata
                if not isinstance(result, dict):
                    result = {'output': result}
                
                # Add standard fields if missing
                result.setdefault('document_id', i)
                result.setdefault('source', doc.metadata.get('source', 'Unknown'))
                result.setdefault('metadata', doc.metadata)
                    
                results.append(result)
            
            return {"results": results}
            
        except Exception as e:
            if "Error executing Python code" in str(e):
                raise  # Re-raise execution errors as-is
            raise NodeExecutionError(f"Node {self.node_id}: Failed to execute Python code: {str(e)}")


class PythonResultProcessorNode(ResultProcessor, PythonCodeMixin):
    """Executes custom Python code on processing results with proper security controls."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = inputs.get("results", [])
        if not results:
            return {"results": []}
        
        # Load Python code
        python_code = self._load_code(self.node_id, self.config)
        
        try:
            processed_results = []
            for i, result in enumerate(results):
                # Set up local variables for this result
                local_vars = {
                    'result': result.copy(),  # Original result (modifiable copy)
                    'original_result': result,  # Read-only reference to original
                    'result_id': i,
                    'processed_result': {}  # For the user to populate
                }
                
                # Execute the user's Python code
                executed_vars = self._execute_code_safely(
                    self.node_id, python_code, local_vars, i, "result"
                )
                
                # Get the processed result
                processed_result = executed_vars.get('processed_result', {})
                
                # If processed_result is empty, use the modified 'result'
                if not processed_result:
                    processed_result = executed_vars.get('result', result)
                
                # Ensure processed_result is a dictionary
                if not isinstance(processed_result, dict):
                    processed_result = {'output': processed_result}
                
                # Add result metadata
                processed_result.setdefault('result_id', i)
                    
                processed_results.append(processed_result)
            
            return {"results": processed_results}
            
        except Exception as e:
            if "Error executing Python code" in str(e):
                raise  # Re-raise execution errors as-is
            raise NodeExecutionError(f"Node {self.node_id}: Failed to execute Python code: {str(e)}")