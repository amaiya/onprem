import os
import sys
import json
import yaml
import tempfile
import streamlit as st
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path to allow importing when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import from parent modules
from OnPrem import read_config
from utils import load_llm

# Import workflow engine
try:
    from onprem.workflow import WorkflowEngine, NODE_REGISTRY
    from onprem.workflow.exceptions import WorkflowValidationError, NodeExecutionError
    WORKFLOW_AVAILABLE = True
except ImportError:
    WORKFLOW_AVAILABLE = False

def get_valid_next_nodes(current_workflow: List[Dict] = None):
    """Get valid next nodes based on current workflow state using workflow engine logic."""
    from onprem.workflow.base import NODE_TYPES
    from onprem.workflow.registry import NODE_REGISTRY
    
    # If no nodes yet, start with Query nodes (since we work with existing vector stores)
    if not current_workflow:
        return ["Query"]
    else:
        # Get the last node in the workflow
        last_node = current_workflow[-1]
        last_node_name = last_node['type']
        
        # Get the actual node class and create a temporary instance to check its NODE_TYPE
        if last_node_name in NODE_REGISTRY:
            last_node_class = NODE_REGISTRY[last_node_name]
            last_node_type = last_node_class.NODE_TYPE
            
            # Get valid connection targets from NODE_TYPES
            node_type_info = NODE_TYPES.get(last_node_type)
            return node_type_info.can_connect_to if node_type_info else []
        else:
            return []

def get_available_nodes(current_workflow: List[Dict] = None):
    """Get nodes that work with existing vector stores, filtered by workflow state."""
    
    # Detect current web app store configuration
    cfg, _ = read_config()
    store_type = cfg.get("llm", {}).get("store_type", "dense")
    
    # Get valid next node types based on workflow state
    valid_node_types = get_valid_next_nodes(current_workflow)
    
    # Define all possible query nodes
    all_query_nodes = {
        'QueryDualStore': {
            'description': 'Query dual vector store (semantic + keyword)',
            'inputs': {},
            'outputs': {'documents': 'List[Document]'},
            'config_fields': {
                'query': {'type': 'text', 'required': True, 'help': 'Search query'},
                'limit': {'type': 'number', 'default': 10, 'help': 'Maximum results'},
                'search_type': {'type': 'select', 'options': ['sparse', 'semantic', 'hybrid'], 'default': 'hybrid'},
                'dense_weight': {'type': 'number', 'default': 0.6, 'help': 'Weight for semantic search (0.0-1.0)'},
                'sparse_weight': {'type': 'number', 'default': 0.4, 'help': 'Weight for sparse search (0.0-1.0)'}
            }
        },
        'QueryWhooshStore': {
            'description': 'Query Whoosh vector store (keyword search)',
            'inputs': {},
            'outputs': {'documents': 'List[Document]'},
            'config_fields': {
                'query': {'type': 'text', 'required': True, 'help': 'Search query'},
                'limit': {'type': 'number', 'default': 10, 'help': 'Maximum results'},
                'search_type': {'type': 'select', 'options': ['sparse', 'semantic'], 'default': 'semantic'}
            }
        },
        'QueryChromaStore': {
            'description': 'Query Chroma vector store (semantic search)',
            'inputs': {},
            'outputs': {'documents': 'List[Document]'},
            'config_fields': {
                'query': {'type': 'text', 'required': True, 'help': 'Search query'},
                'limit': {'type': 'number', 'default': 10, 'help': 'Maximum results'}
            }
        },
        'QueryElasticsearchStore': {
            'description': 'Query Elasticsearch vector store (hybrid search)',
            'inputs': {},
            'outputs': {'documents': 'List[Document]'},
            'config_fields': {
                'query': {'type': 'text', 'required': True, 'help': 'Search query'},
                'limit': {'type': 'number', 'default': 10, 'help': 'Maximum results'},
                'search_type': {'type': 'select', 'options': ['sparse', 'semantic', 'hybrid'], 'default': 'hybrid'}
            }
        }
    }
    
    # Filter query nodes based on web app store configuration
    query_nodes = {}
    if store_type == "dual":
        query_nodes['QueryDualStore'] = all_query_nodes['QueryDualStore']
    elif store_type == "sparse":
        query_nodes['QueryWhooshStore'] = all_query_nodes['QueryWhooshStore']
    elif store_type in ["dense", "chroma"]:
        query_nodes['QueryChromaStore'] = all_query_nodes['QueryChromaStore']
    elif store_type == "elasticsearch":
        query_nodes['QueryElasticsearchStore'] = all_query_nodes['QueryElasticsearchStore']
    else:
        # Default fallback to dual if unknown store type
        query_nodes['QueryDualStore'] = all_query_nodes['QueryDualStore']
    
    all_nodes = {
        # Query nodes (starting points) - filtered by web app configuration
        'Query': query_nodes,
        # Document transformers
        'Document Transformers': {
            'AddMetadata': {
                'description': 'Add custom metadata fields to documents',
                'inputs': {'documents': 'List[Document]'},
                'outputs': {'documents': 'List[Document]'},
                'config_fields': {
                    'metadata': {'type': 'json', 'required': True, 'help': 'JSON object with metadata to add (e.g., {"category": "research", "priority": "high"})'}
                }
            },
            'ContentPrefix': {
                'description': 'Add text to the beginning of document content',
                'inputs': {'documents': 'List[Document]'},
                'outputs': {'documents': 'List[Document]'},
                'config_fields': {
                    'prefix': {'type': 'textarea', 'required': True, 'help': 'Text to prepend to each document'},
                    'separator': {'type': 'text', 'default': '\n\n', 'help': 'Separator between prefix and content'}
                }
            },
            'ContentSuffix': {
                'description': 'Add text to the end of document content',
                'inputs': {'documents': 'List[Document]'},
                'outputs': {'documents': 'List[Document]'},
                'config_fields': {
                    'suffix': {'type': 'textarea', 'required': True, 'help': 'Text to append to each document'},
                    'separator': {'type': 'text', 'default': '\n\n', 'help': 'Separator between content and suffix'}
                }
            },
            'DocumentFilter': {
                'description': 'Filter documents by content or metadata criteria',
                'inputs': {'documents': 'List[Document]'},
                'outputs': {'documents': 'List[Document]'},
                'config_fields': {
                    'content_contains': {'type': 'text', 'help': 'Comma-separated terms that content must contain (any)'},
                    'content_excludes': {'type': 'text', 'help': 'Comma-separated terms that content must not contain (any)'},
                    'min_length': {'type': 'number', 'help': 'Minimum content length in characters'},
                    'max_length': {'type': 'number', 'help': 'Maximum content length in characters'},
                    'metadata_filters': {'type': 'json', 'help': 'JSON object with metadata filters (e.g., {"category": "research"})'}
                }
            },
            'PythonDocumentTransformer': {
                'description': 'Transform documents using custom Python code',
                'inputs': {'documents': 'List[Document]'},
                'outputs': {'documents': 'List[Document]'},
                'config_fields': {
                    'code': {
                        'type': 'textarea', 
                        'required': True, 
                        'help': 'Python code to transform documents. Available variables: doc, content, metadata, document_id, source',
                        'default': '''# PythonDocumentTransformer: Modify document content and metadata
# Available variables: doc, content, metadata, document_id, source
# Output: Modified doc object (transforms documents in place)

#----------------
# EXAMPLE
#----------------
# Clean up content
import re
content = re.sub(r'\\s+', ' ', content).strip()

# Add metadata
metadata['word_count'] = len(content.split())
metadata['doc_type'] = 'email' if '@' in content else 'general'

# Update document content
doc.page_content = content'''
                    }
                }
            }
        },
        # Document processors
        'Processors': {
            'PromptProcessor': {
                'description': 'Apply LLM prompt to each document in search results',
                'inputs': {'documents': 'List[Document]'},
                'outputs': {'results': 'List[Dict]'},
                'config_fields': {
                    'prompt': {'type': 'textarea', 'required': True, 'help': 'LLM prompt template'},
                    'batch_size': {'type': 'number', 'default': 5, 'help': 'Documents per batch'}
                }
            },
            'SummaryProcessor': {
                'description': 'Generate document summaries for each document in search results',
                'inputs': {'documents': 'List[Document]'},
                'outputs': {'results': 'List[Dict]'},
                'config_fields': {
                    'batch_size': {'type': 'number', 'default': 5, 'help': 'Documents per batch'}
                }
            },
            'ResponseCleaner': {
                'description': 'Clean and refine LLM responses (e.g., place after PromptProcessor)',
                'inputs': {'results': 'List[Dict]'},
                'outputs': {'results': 'List[Dict]'},
                'config_fields': {
                    'cleanup_prompt': {'type': 'textarea', 'required': True, 'help': 'Cleanup prompt template'}
                }
            },
            'DocumentToResults': {
                'description': 'Directly convert documents to export-ready results (without AI analysis)',
                'inputs': {'documents': 'List[Document]'},
                'outputs': {'results': 'List[Dict]'},
                'config_fields': {
                    'include_content': {'type': 'checkbox', 'default': True, 'help': 'Include document content'},
                    'content_field': {'type': 'text', 'default': 'page_content', 'help': 'Name for content field'},
                    'metadata_prefix': {'type': 'text', 'default': 'meta_', 'help': 'Prefix for metadata fields'}
                }
            },
            'PythonDocumentProcessor': {
                'description': 'Process documents using custom Python code (returns results format)',
                'inputs': {'documents': 'List[Document]'},
                'outputs': {'results': 'List[Dict]'},
                'config_fields': {
                    'code': {
                        'type': 'textarea', 
                        'required': True, 
                        'help': 'Python code to process documents. Available variables: doc, content, metadata, document_id, source. Set result_dict for each document.',
                        'default': '''# PythonDocumentProcessor: Convert documents to structured results
# Available variables: doc, content, metadata, document_id, source
# Output: Set result_dict with extracted information

#----------------
# EXAMPLE
#----------------

import re

# Extract information
emails = re.findall(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', content)

# Set result dictionary for this document
result_dict = {
    'source': source,
    'text_length': len(content),
    'word_count': len(content.split()),
    'has_email': len(emails) > 0,
    'preview': content[:100] + '...' if len(content) > 100 else content
}'''
                    }
                }
            },
            'PythonResultProcessor': {
                'description': 'Process results using custom Python code',
                'inputs': {'results': 'List[Dict]'},
                'outputs': {'results': 'List[Dict]'},
                'config_fields': {
                    'code': {
                        'type': 'textarea', 
                        'required': True, 
                        'help': 'Python code to process results. Available variables: result, result_id. Modify result dict in place or set new_result.',
                        'default': '''# PythonResultProcessor: Enhance and post-process results
# Available variables: result (dict), result_id
# Output: Modify result dict in place or set new_result


#----------------
# EXAMPLE
#----------------

import datetime
import os

# Add timestamp and metadata
result['processed_at'] = datetime.datetime.now().isoformat()

# Add confidence score based on text length
if 'text_length' in result:
    result['confidence'] = 'high' if result['text_length'] > 1000 else 'low'

# Extract filename from source
if 'source' in result:
    result['filename'] = os.path.basename(result['source'])'''
                    }
                }
            }
        },
        # Aggregators
        'Aggregators': {
            'AggregatorNode': {
                'description': 'Aggregate multiple results with LLM',
                'inputs': {'results': 'List[Dict]'},
                'outputs': {'result': 'Dict'},
                'config_fields': {
                    'prompt': {'type': 'textarea', 'required': True, 'help': 'Aggregation prompt template'}
                }
            },
            'PythonAggregatorNode': {
                'description': 'Aggregate results with Python code',
                'inputs': {'results': 'List[Dict]'},
                'outputs': {'result': 'Dict'},
                'config_fields': {
                    'code': {
                        'type': 'textarea', 
                        'required': True, 
                        'help': 'Python aggregation code. Available variables: results (List[Dict]). Set aggregated_result dict.',
                        'default': '''# PythonAggregatorNode: Combine multiple results into single summary
# Available variables: results (List[Dict])
# Output: Set aggregated_result dict

#----------------
# EXAMPLE
#----------------

import datetime

# Create aggregated summary
aggregated_result = {
    'timestamp': datetime.datetime.now().isoformat(),
    'total_results': len(results),
    'total_length': sum(r.get('text_length', 0) for r in results),
    'sources': [r.get('source', 'Unknown') for r in results],
    'high_confidence_count': sum(1 for r in results if r.get('confidence') == 'high')
}'''
                    }
                }
            }
        },
        # Exporters
        'Exporters': {
            'CSVExporter': {
                'description': 'Export results to CSV',
                'inputs': {'results': 'List[Dict]'},
                'outputs': {'status': 'str'},
                'config_fields': {
                    'output_path': {'type': 'text', 'default': 'results.csv', 'help': 'Output file path'}
                }
            },
            'ExcelExporter': {
                'description': 'Export results to Excel spreadsheet',
                'inputs': {'results': 'List[Dict]'},
                'outputs': {'status': 'str'},
                'config_fields': {
                    'output_path': {'type': 'text', 'default': 'results.xlsx', 'help': 'Output file path'},
                    'sheet_name': {'type': 'text', 'default': 'Results', 'help': 'Name of the Excel sheet'}
                }
            },
            'JSONExporter': {
                'description': 'Export results to JSON',
                'inputs': {'results': 'List[Dict]', 'result': 'Dict'},
                'outputs': {'status': 'str'},
                'config_fields': {
                    'output_path': {'type': 'text', 'default': 'results.json', 'help': 'Output file path'},
                    'pretty_print': {'type': 'checkbox', 'default': True, 'help': 'Format JSON nicely'}
                }
            },
            'JSONResponseExporter': {
                'description': 'Extract and export JSON from responses',
                'inputs': {'results': 'List[Dict]', 'result': 'Dict'},
                'outputs': {'status': 'str'},
                'config_fields': {
                    'output_path': {'type': 'text', 'default': 'extracted_responses.json', 'help': 'Output file path'},
                    'response_field': {'type': 'text', 'default': 'response', 'help': 'Field containing JSON response'}
                }
            }
        }
    }
    
    # Filter nodes based on valid next node types using workflow engine logic
    from onprem.workflow.registry import NODE_REGISTRY
    
    filtered_categories = {}
    
    # Go through each category
    for category_name, category_nodes in all_nodes.items():
        filtered_category_nodes = {}
        
        # Filter nodes within this category
        for node_name, node_config in category_nodes.items():
            # Get the node class and check its NODE_TYPE
            if node_name in NODE_REGISTRY:
                node_class = NODE_REGISTRY[node_name]
                node_type = node_class.NODE_TYPE
                
                # Only include if this node type is valid for the current workflow state
                if node_type in valid_node_types:
                    filtered_category_nodes[node_name] = node_config
        
        # Only include category if it has any valid nodes
        if filtered_category_nodes:
            filtered_categories[category_name] = filtered_category_nodes
    
    return filtered_categories

def render_node_config(node_type: str, node_data: Dict, node_id: str) -> Dict:
    """Render configuration UI for a node and return config"""
    config = {}
    
    st.subheader(f"{node_type} Configuration")
    st.write(f"**{node_id}**: {node_data['description']}")
    
    # Add LLM config for processor nodes
    if node_type in ['PromptProcessor', 'SummaryProcessor', 'ResponseCleaner', 'AggregatorNode']:
        st.write("**LLM Configuration**")
        model_url = st.text_input("Model URL", value="openai://gpt-3.5-turbo", key=f"{node_id}_model_url",
                                help="LLM model URL (e.g., openai://gpt-4, anthropic://claude-3-sonnet)")
        config['llm'] = {'model_url': model_url}
    
    # Add vector store path for query nodes
    if node_type.startswith('Query'):
        cfg, _ = read_config()
        vectordb_path = cfg.get('vectordb_path', '~/onprem_data/webapp/vectordb')
        config['persist_location'] = st.text_input("Vector Store Path", value=vectordb_path, 
                                                  key=f"{node_id}_vectordb_path",
                                                  help="Path to existing vector store")
    
    # Render node-specific config fields
    for field_name, field_config in node_data['config_fields'].items():
        key = f"{node_id}_{field_name}"
        
        if field_config['type'] == 'text':
            value = st.text_input(
                field_name.replace('_', ' ').title(),
                value=field_config.get('default', ''),
                key=key,
                help=field_config.get('help', '')
            )
            if value or field_config.get('required'):
                config[field_name] = value
                
        elif field_config['type'] == 'textarea':
            value = st.text_area(
                field_name.replace('_', ' ').title(),
                value=field_config.get('default', ''),
                height=150,
                key=key,
                help=field_config.get('help', '')
            )
            if value or field_config.get('required'):
                config[field_name] = value
                
        elif field_config['type'] == 'number':
            # Special handling for weights (0.0-1.0 range)
            if 'weight' in field_name:
                value = st.slider(
                    field_name.replace('_', ' ').title(),
                    min_value=0.0,
                    max_value=1.0,
                    value=field_config.get('default', 0.5),
                    step=0.1,
                    key=key,
                    help=field_config.get('help', '')
                )
            else:
                value = st.number_input(
                    field_name.replace('_', ' ').title(),
                    value=field_config.get('default', 1),
                    min_value=1,
                    key=key,
                    help=field_config.get('help', '')
                )
            config[field_name] = value
            
        elif field_config['type'] == 'checkbox':
            value = st.checkbox(
                field_name.replace('_', ' ').title(),
                value=field_config.get('default', False),
                key=key,
                help=field_config.get('help', '')
            )
            config[field_name] = value
            
        elif field_config['type'] == 'select':
            value = st.selectbox(
                field_name.replace('_', ' ').title(),
                options=field_config['options'],
                index=field_config['options'].index(field_config.get('default', field_config['options'][0])),
                key=key,
                help=field_config.get('help', '')
            )
            config[field_name] = value
            
        elif field_config['type'] == 'json':
            import json
            placeholder_text = field_config.get('help', 'Enter JSON object')
            default_text = field_config.get('default', '{}')
            
            value = st.text_area(
                field_name.replace('_', ' ').title(),
                value=default_text if isinstance(default_text, str) else json.dumps(default_text, indent=2),
                height=100,
                key=key,
                help=placeholder_text,
                placeholder='{"key": "value", "number": 123}'
            )
            
            # Validate and parse JSON
            if value.strip():
                try:
                    parsed_value = json.loads(value)
                    config[field_name] = parsed_value
                    # Clear any previous error state
                    if f"{key}_error" in st.session_state:
                        del st.session_state[f"{key}_error"]
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON in {field_name}: {str(e)}")
                    st.session_state[f"{key}_error"] = True
                    # Don't include invalid JSON in config
            elif field_config.get('required'):
                st.error(f"{field_name} is required")
    
    return config

def generate_workflow_yaml(workflow_nodes: List[Dict]) -> str:
    """Generate YAML workflow from node configuration"""
    if not workflow_nodes:
        return "# No nodes configured"
    
    nodes = {}
    connections = []
    
    # Build nodes
    for i, node in enumerate(workflow_nodes):
        node_id = node['id']
        config = node['config'].copy()
        
        # Special handling for QueryDualStore weights
        if node['type'] == 'QueryDualStore' and 'dense_weight' in config and 'sparse_weight' in config:
            weights = [config['dense_weight'], config['sparse_weight']]
            config['weights'] = weights
            # Remove individual weight fields
            config.pop('dense_weight', None)
            config.pop('sparse_weight', None)
        
        # Special handling for DocumentFilter comma-separated lists
        if node['type'] == 'DocumentFilter':
            # Convert comma-separated strings to lists
            for field_name in ['content_contains', 'content_excludes']:
                if field_name in config and isinstance(config[field_name], str):
                    # Split by comma and strip whitespace
                    config[field_name] = [term.strip() for term in config[field_name].split(',') if term.strip()]
        
        nodes[node_id] = {
            'type': node['type'],
            'config': config
        }
        
        # Add connections (simple linear chain for now)
        if i > 0:
            prev_node = workflow_nodes[i-1]
            # Determine correct port connections based on node types
            from_port = get_output_port(prev_node['type'])
            to_port = get_input_port(node['type'])
            
            connections.append({
                'from': prev_node['id'],
                'from_port': from_port,
                'to': node_id,
                'to_port': to_port
            })
    
    workflow = {
        'nodes': nodes,
        'connections': connections
    }
    
    return yaml.dump(workflow, default_flow_style=False, sort_keys=False)

def get_output_port(node_type: str) -> str:
    """Get the primary output port for a node type"""
    if node_type.startswith('Query'):
        return 'documents'
    elif node_type in ['PromptProcessor', 'SummaryProcessor', 'ResponseCleaner', 'PythonDocumentProcessor', 'PythonResultProcessor', 'DocumentToResults']:
        return 'results'
    elif node_type in ['AggregatorNode', 'PythonAggregatorNode']:
        return 'result'
    else:
        return 'status'

def get_input_port(node_type: str) -> str:
    """Get the primary input port for a node type"""
    if node_type in ['PromptProcessor', 'SummaryProcessor', 'PythonDocumentProcessor', 'DocumentToResults']:
        return 'documents'
    elif node_type in ['ResponseCleaner', 'PythonResultProcessor']:
        return 'results'
    elif node_type in ['AggregatorNode', 'PythonAggregatorNode']:
        return 'results'
    elif node_type.endswith('Exporter'):
        return 'results'
    else:
        return 'documents'

def main():
    """Visual Workflow Builder main function"""
    if not WORKFLOW_AVAILABLE:
        st.error("⚠️ Workflow engine not available. Please install the workflow dependencies.")
        return
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .workflow-node {
        border: 2px solid #e1e5e9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    .node-query { border-color: #007bff; background-color: #e7f3ff; }
    .node-processor { border-color: #28a745; background-color: #e7f8ea; }
    .node-aggregator { border-color: #ffc107; background-color: #fff8e1; }
    .node-exporter { border-color: #dc3545; background-color: #ffeaea; }
    
    .workflow-chain {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1rem 0;
        padding: 1rem;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 8px;
        overflow-x: auto;
    }
    .node-box {
        min-width: 100px;
        padding: 0.5rem;
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        text-align: center;
        font-size: 0.75rem;
        white-space: nowrap;
    }
    .workflow-arrow {
        font-size: 1.2rem;
        color: #6c757d;
        margin: 0 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <h1 style="
        color: #0068c9;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0068c9;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
    ">
        <span style="font-size: 1.8rem; margin-right: 0.5rem;">🔧</span> 
        Visual Workflow Builder
    </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Build workflows visually using your existing vector store. All workflows start with a query node 
    to search your documents, then apply processing, aggregation, and export steps.
    
    💡 **Quick Start:** Add a Query node → Processor node → Exporter node for a complete workflow.
    """)
    
    # Initialize session state
    if 'workflow_nodes' not in st.session_state:
        st.session_state.workflow_nodes = []
    if 'node_counter' not in st.session_state:
        st.session_state.node_counter = 1
    
    # Get available nodes filtered by current workflow state
    available_nodes = get_available_nodes(st.session_state.workflow_nodes)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Add Nodes")
        
        if not available_nodes:
            st.info("✅ Workflow is complete! No additional nodes can be added.")
            st.write("Exporter nodes are terminal - they cannot connect to other nodes.")
        else:
            # Node selection
            selected_category = st.selectbox("Node Category", list(available_nodes.keys()))
            selected_node_type = st.selectbox(
                "Node Type", 
                list(available_nodes[selected_category].keys())
            )
            
            # Show node description
            node_data = available_nodes[selected_category][selected_node_type]
            st.info(f"**Description:** {node_data['description']}")
            
            # Add node button
            if st.button("➕ Add Node"):
                # Validation: Ensure first node is a Query node
                if not st.session_state.workflow_nodes and selected_category != 'Query':
                    st.error("⚠️ First node must be a Query node to search your documents!")
                else:
                    node_id = f"node_{st.session_state.node_counter}"
                    st.session_state.node_counter += 1
                    
                    # Add to workflow
                    st.session_state.workflow_nodes.append({
                        'id': node_id,
                        'type': selected_node_type,
                        'category': selected_category,
                        'data': node_data,
                        'config': {}
                    })
                    st.success(f"✅ Added {selected_node_type} node!")
                    st.rerun()
        
        # Clear workflow button
        if st.button("🗑️ Clear Workflow"):
            st.session_state.workflow_nodes = []
            st.session_state.node_counter = 1
            st.rerun()
        
        st.markdown("---")
        
        # Quick templates
        st.subheader("📋 Quick Templates")
        
        if st.button("🚀 Document Analysis Template"):
            st.session_state.workflow_nodes = [
                {
                    'id': 'query_docs',
                    'type': 'QueryDualStore',
                    'category': 'Query',
                    'data': available_nodes['Query']['QueryDualStore'],
                    'config': {'query': 'artificial intelligence', 'limit': 10, 'search_type': 'hybrid'}
                },
                {
                    'id': 'analyze_docs',
                    'type': 'PromptProcessor',
                    'category': 'Processors',
                    'data': available_nodes['Processors']['PromptProcessor'],
                    'config': {'prompt': 'Analyze this document and extract key themes:\n\n{content}'}
                },
                {
                    'id': 'export_results',
                    'type': 'CSVExporter',
                    'category': 'Exporters',
                    'data': available_nodes['Exporters']['CSVExporter'],
                    'config': {'output_path': 'document_analysis.csv'}
                }
            ]
            st.session_state.node_counter = 4
            st.rerun()
        
        if st.button("📊 Summary & Aggregate Template"):
            st.session_state.workflow_nodes = [
                {
                    'id': 'search_docs',
                    'type': 'QueryDualStore',
                    'category': 'Query',
                    'data': available_nodes['Query']['QueryDualStore'],
                    'config': {'query': 'project report', 'limit': 5, 'search_type': 'semantic'}
                },
                {
                    'id': 'summarize',
                    'type': 'SummaryProcessor',
                    'category': 'Processors',
                    'data': available_nodes['Processors']['SummaryProcessor'],
                    'config': {}
                },
                {
                    'id': 'aggregate_summaries',
                    'type': 'AggregatorNode',
                    'category': 'Aggregators',
                    'data': available_nodes['Aggregators']['AggregatorNode'],
                    'config': {'prompt': 'Combine these summaries into a comprehensive overview:\n\n{responses}'}
                },
                {
                    'id': 'export_json',
                    'type': 'JSONExporter',
                    'category': 'Exporters',
                    'data': available_nodes['Exporters']['JSONExporter'],
                    'config': {'output_path': 'aggregated_summary.json'}
                }
            ]
            st.session_state.node_counter = 5
            st.rerun()
    
    with col2:
        st.header("Workflow Configuration")
        
        if not st.session_state.workflow_nodes:
            st.info("👆 Add nodes from the left panel to start building your workflow")
        else:
            ## Show workflow chain
            #st.subheader("Workflow Chain")
            
            # Create visual workflow chain
            chain_html = '<div class="workflow-chain">'
            for i, node in enumerate(st.session_state.workflow_nodes):
                if i > 0:
                    chain_html += '<span class="workflow-arrow">→</span>'
                
                # Determine node color class
                if node['category'] == 'Query':
                    color_class = 'node-query'
                elif node['category'] == 'Processors':
                    color_class = 'node-processor'
                elif node['category'] == 'Aggregators':
                    color_class = 'node-aggregator'
                else:
                    color_class = 'node-exporter'
                
                chain_html += f'''
                <div class="node-box {color_class}">
                    <strong>{node['id']}</strong><br>
                    {node['type']}
                </div>
                '''
            chain_html += '</div>'
            
            st.markdown(chain_html, unsafe_allow_html=True)
            
            # Configure each node
            tabs = st.tabs([f"{node['id']}" for node in st.session_state.workflow_nodes])
            
            for i, (tab, node) in enumerate(zip(tabs, st.session_state.workflow_nodes)):
                with tab:
                    # Node configuration
                    config = render_node_config(node['type'], node['data'], node['id'])
                    st.session_state.workflow_nodes[i]['config'] = config
                    
                    # Remove node button
                    if st.button(f"🗑️ Remove {node['id']}", key=f"remove_{node['id']}"):
                        st.session_state.workflow_nodes.pop(i)
                        st.rerun()
    
    # Generate and run workflow
    if st.session_state.workflow_nodes:
        st.header("Generated Workflow")
        
        # Generate YAML
        workflow_yaml = generate_workflow_yaml(st.session_state.workflow_nodes)
        
        # Show YAML in expandable section
        with st.expander("📄 View Generated YAML"):
            st.code(workflow_yaml, language='yaml')
        
        # Download YAML
        st.download_button(
            "💾 Download Workflow YAML",
            workflow_yaml,
            file_name="workflow.yaml",
            mime="text/yaml"
        )
        
        # Run workflow
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Run Workflow", type="primary"):
                try:
                    with st.spinner("Running workflow..."):
                        # Parse YAML and execute
                        workflow_dict = yaml.safe_load(workflow_yaml)
                        
                        engine = WorkflowEngine()
                        engine.load_workflow_from_dict(workflow_dict)
                        results = engine.execute()
                        
                        st.success("✅ Workflow completed successfully!")
                        
                        # Show results
                        st.subheader("Results")
                        for node_id, node_results in results.items():
                            with st.expander(f"📊 {node_id} Results"):
                                if isinstance(node_results, dict):
                                    if 'status' in node_results:
                                        st.info(f"Status: {node_results['status']}")
                                    else:
                                        st.json(node_results)
                                else:
                                    st.write(node_results)
                        
                except Exception as e:
                    st.error(f"❌ Workflow execution failed: {str(e)}")
                    st.exception(e)
        
        with col2:
            if st.button("✅ Validate Workflow"):
                try:
                    workflow_dict = yaml.safe_load(workflow_yaml)
                    engine = WorkflowEngine()
                    engine.load_workflow_from_dict(workflow_dict)
                    st.success("✅ Workflow is valid!")
                except WorkflowValidationError as e:
                    st.error(f"❌ Validation failed: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
