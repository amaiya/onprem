"""
Processor node implementations for workflow pipelines.

This module contains processor nodes that apply various types of processing
to documents or results using LLMs or other processing methods.
"""

import os
from typing import Dict, List, Any
from langchain_core.documents import Document

from .base import DocumentProcessor, ResultProcessor, AggregatorProcessor
from .exceptions import NodeExecutionError
from ..utils import format_string


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
                
                formatted_prompt = format_string(prompt_template, **format_kwargs)
                
                # Get LLM response
                response = llm.prompt(formatted_prompt)
                
                # Create result record
                result = {
                    'document_id': i,
                    'content': doc.page_content,
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
        source_field = self.config.get("source_field", "response")  # Field to clean
        
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
                # Get content from the specified source field
                original_content = result.get(source_field, '')
                
                if not original_content:
                    # If specified field is empty, skip this result or copy as-is
                    cleaned_results.append(result.copy())
                    continue
                
                # Format cleanup prompt with the original content
                format_kwargs = {
                    'original_response': original_content,
                    'response': original_content,  # Alias for backward compatibility
                    'content': original_content,   # Another common alias
                    **result  # Include all result fields for additional context
                }
                cleanup_request = format_string(cleanup_prompt_template, **format_kwargs)
                
                # Get cleaned response
                cleaned_response = llm.prompt(cleanup_request)
                
                # Create new result with cleaned response
                cleaned_result = result.copy()
                cleaned_result[source_field] = cleaned_response.strip()  # Update the source field
                cleaned_result[f'original_{source_field}'] = original_content  # Keep original for reference
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
                    'content' : doc.page_content,
                    'original_length': len(doc.page_content),
                    'response': summary,
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


class AggregatorNode(AggregatorProcessor):
    """Aggregates multiple results into a single result using LLM-based processing."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = inputs.get("results", [])
        if not results:
            return {"result": {}}
        
        prompt_template = self.config.get("prompt", "")
        prompt_file = self.config.get("prompt_file", "")
        
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
            
            # Prepare aggregation data
            responses = []
            for i, result in enumerate(results):
                # Extract the main response/content from each result
                response = result.get('response', result.get('summary', result.get('output', str(result))))
                responses.append(response)
            
            # Create context for the aggregation prompt
            aggregation_context = {
                'num_results': len(results),
                'responses': '\n\n'.join([f"Response {i+1}: {resp}" for i, resp in enumerate(responses)]),
                'results': results,  # Full results for advanced templating
                'response_list': responses  # Just the responses as a list
            }
            
            # Format the prompt
            formatted_prompt = format_string(prompt_template, **aggregation_context)
            
            # Get aggregated response from LLM
            aggregated_response = llm.prompt(formatted_prompt)
            
            # Create aggregated result
            aggregated_result = {
                'aggregated_response': aggregated_response,
                'source_count': len(results),
                'aggregation_method': 'llm_prompt',
                #'original_results': results  # Keep reference to source data
            }
            
            return {"result": aggregated_result}
            
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to aggregate results: {str(e)}")


class PythonAggregatorNode(AggregatorProcessor, PythonCodeMixin):
    """Aggregates multiple results into a single result using custom Python code."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = inputs.get("results", [])
        if not results:
            return {"result": {}}
        
        # Load Python code
        python_code = self._load_code(self.node_id, self.config)
        
        try:
            # Set up local variables for aggregation
            local_vars = {
                'results': results,
                'num_results': len(results),
                'result': {}  # For the user to populate with aggregated result
            }
            
            # Execute the user's Python code
            executed_vars = self._execute_code_safely(
                self.node_id, python_code, local_vars, 0, "aggregation"
            )
            
            # Get the aggregated result
            aggregated_result = executed_vars.get('result', {})
            
            # Ensure result is a dictionary
            if not isinstance(aggregated_result, dict):
                aggregated_result = {'output': aggregated_result}
            
            # Add metadata
            aggregated_result.setdefault('source_count', len(results))
            aggregated_result.setdefault('aggregation_method', 'python_code')
            
            return {"result": aggregated_result}
            
        except Exception as e:
            if "Error executing Python code" in str(e):
                raise  # Re-raise execution errors as-is
            raise NodeExecutionError(f"Node {self.node_id}: Failed to execute aggregation code: {str(e)}")


class DocumentToResultsNode(DocumentProcessor):
    """Converts List[Document] to List[Dict] format suitable for Exporter nodes.
    
    This node bridges Query nodes (which output documents) to Exporter nodes 
    (which expect processing results). It converts each document into a dictionary
    with content, metadata, and optional custom fields.
    """
    
    def get_input_types(self) -> Dict[str, str]:
        """Override to specify input types."""
        return {"documents": "List[Document]"}
    
    def get_output_types(self) -> Dict[str, str]:
        """Override to output List[Dict] instead of List[Document]."""
        return {"results": "List[Dict]"}
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        documents = inputs.get("documents", [])
        if not documents:
            return {"results": []}
        
        # Configuration options
        include_content = self.config.get("include_content", True)
        include_metadata = self.config.get("include_metadata", True)
        content_field = self.config.get("content_field", "page_content")
        metadata_prefix = self.config.get("metadata_prefix", "meta_")
        flatten_metadata = self.config.get("flatten_metadata", True)
        custom_fields = self.config.get("custom_fields", {})
        
        try:
            results = []
            
            for i, doc in enumerate(documents):
                result = {
                    "document_id": i,
                    "source": doc.metadata.get("source", "unknown")
                }
                
                # Add content if requested
                if include_content:
                    result[content_field] = doc.page_content
                    result["content_length"] = len(doc.page_content)
                
                # Add metadata if requested
                if include_metadata:
                    if flatten_metadata:
                        # Flatten metadata with prefix (compatible with existing exporters)
                        for key, value in doc.metadata.items():
                            # Skip source since we already added it
                            if key != "source":
                                result[f"{metadata_prefix}{key}"] = value
                    else:
                        # Keep metadata as nested object
                        result["metadata"] = doc.metadata
                
                # Add any custom static fields
                result.update(custom_fields)
                
                results.append(result)
            
            return {"results": results}
            
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to convert documents: {str(e)}")
