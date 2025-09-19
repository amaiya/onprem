#!/usr/bin/env python3
"""
Quick test for all workflow node types - creates minimal workflows and cleans up.
"""
import os
import tempfile
import shutil
import pytest
from pathlib import Path

# Test configuration
TEST_TEXT_CONTENT = "This is a test document for workflow testing."
MOCK_LLM_RESPONSE = "Test analysis complete"

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def sample_doc_file(temp_dir):
    """Create a sample text document."""
    doc_path = Path(temp_dir) / "test_doc.txt"
    doc_path.write_text(TEST_TEXT_CONTENT)
    return str(doc_path)

def test_document_transformer_nodes(temp_dir, sample_doc_file):
    """Test all document transformer node types."""
    from onprem.workflow import WorkflowEngine
    
    # Test AddMetadata node
    workflow = {
        "nodes": {
            "loader": {
                "type": "LoadFromFolder",
                "config": {
                    "source_directory": temp_dir,
                    "include_patterns": ["*.txt"]
                }
            },
            "add_metadata": {
                "type": "AddMetadata", 
                "config": {
                    "metadata": {
                        "category": "meeting20251001",
                        "department": "engineering",
                        "priority": "high"
                    }
                }
            }
        },
        "connections": [
            {
                "from": "loader",
                "from_port": "documents",
                "to": "add_metadata",
                "to_port": "documents"
            }
        ]
    }
    
    engine = WorkflowEngine()
    engine.load_workflow_from_dict(workflow)
    results = engine.execute()
    
    # Verify metadata was added
    documents = results["add_metadata"]["documents"]
    assert len(documents) > 0
    doc = documents[0]
    assert doc.metadata.get("category") == "meeting20251001"
    assert doc.metadata.get("department") == "engineering"
    assert doc.metadata.get("priority") == "high"
    
    # Test ContentPrefix node
    workflow_prefix = {
        "nodes": {
            "loader": {
                "type": "LoadFromFolder",
                "config": {
                    "source_directory": temp_dir,
                    "include_patterns": ["*.txt"]
                }
            },
            "add_prefix": {
                "type": "ContentPrefix",
                "config": {
                    "prefix": "[CONFIDENTIAL]",
                    "separator": " "
                }
            }
        },
        "connections": [
            {
                "from": "loader",
                "from_port": "documents", 
                "to": "add_prefix",
                "to_port": "documents"
            }
        ]
    }
    
    engine = WorkflowEngine()
    engine.load_workflow_from_dict(workflow_prefix)
    results = engine.execute()
    
    # Verify prefix was added
    documents = results["add_prefix"]["documents"]
    assert len(documents) > 0
    assert documents[0].page_content.startswith("[CONFIDENTIAL] ")
    
    # Test DocumentFilter node
    workflow_filter = {
        "nodes": {
            "loader": {
                "type": "LoadFromFolder",
                "config": {
                    "source_directory": temp_dir,
                    "include_patterns": ["*.txt"]
                }
            },
            "filter_docs": {
                "type": "DocumentFilter",
                "config": {
                    "content_contains": ["test"],
                    "min_length": 10
                }
            }
        },
        "connections": [
            {
                "from": "loader",
                "from_port": "documents",
                "to": "filter_docs", 
                "to_port": "documents"
            }
        ]
    }
    
    engine = WorkflowEngine()
    engine.load_workflow_from_dict(workflow_filter)
    results = engine.execute()
    
    # Verify filtering worked
    documents = results["filter_docs"]["documents"]
    assert len(documents) > 0  # Should pass filter (contains "test" and > 10 chars)
    
    # Test PythonDocumentTransformer node
    workflow_python = {
        "nodes": {
            "loader": {
                "type": "LoadFromFolder", 
                "config": {
                    "source_directory": temp_dir,
                    "include_patterns": ["*.txt"]
                }
            },
            "python_transform": {
                "type": "PythonDocumentTransformer",
                "config": {
                    "code": """
# Add word count to metadata and uppercase content
word_count = len(content.split())
metadata['word_count'] = word_count
metadata['processed_by'] = 'python_transformer'

# Transform content to uppercase
content = content.upper()

# Set the transformed document
transformed_doc = Document(
    page_content=content,
    metadata=metadata
)
"""
                }
            }
        },
        "connections": [
            {
                "from": "loader",
                "from_port": "documents",
                "to": "python_transform",
                "to_port": "documents"
            }
        ]
    }
    
    engine = WorkflowEngine()
    engine.load_workflow_from_dict(workflow_python)
    results = engine.execute()
    
    # Verify Python transformation worked
    documents = results["python_transform"]["documents"]
    assert len(documents) > 0
    doc = documents[0]
    assert doc.page_content == TEST_TEXT_CONTENT.upper()
    assert doc.metadata.get("word_count") == len(TEST_TEXT_CONTENT.split())
    assert doc.metadata.get("processed_by") == "python_transformer"


def test_loader_nodes(temp_dir, sample_doc_file):
    """Test all loader node types."""
    from onprem.workflow import WorkflowEngine
    
    # Test LoadFromFolder
    workflow = {
        "nodes": {
            "loader": {
                "type": "LoadFromFolder",
                "config": {
                    "source_directory": temp_dir,
                    "include_patterns": ["*.txt"]
                }
            }
        },
        "connections": []
    }
    
    engine = WorkflowEngine()
    engine.load_workflow_from_dict(workflow)
    results = engine.execute()
    assert "loader" in results
    assert len(results["loader"]["documents"]) > 0
    print("✓ LoadFromFolder")
    
    # Test LoadSingleDocument  
    workflow = {
        "nodes": {
            "single_loader": {
                "type": "LoadSingleDocument", 
                "config": {
                    "file_path": sample_doc_file
                }
            }
        },
        "connections": []
    }
    
    engine = WorkflowEngine()
    engine.load_workflow_from_dict(workflow)
    results = engine.execute()
    assert "single_loader" in results
    assert len(results["single_loader"]["documents"]) == 1
    print("✓ LoadSingleDocument")

def test_spreadsheet_loader(temp_dir):
    """Test LoadSpreadsheet node."""
    from onprem.workflow import WorkflowEngine
    import csv
    
    # Create test CSV file
    csv_path = Path(temp_dir) / "test_data.csv"
    test_data = [
        {"id": 1, "text": "First document text", "category": "A", "priority": "high"},
        {"id": 2, "text": "Second document content", "category": "B", "priority": "low"},
        {"id": 3, "text": "Third document info", "category": "A", "priority": "medium"}
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "text", "category", "priority"])
        writer.writeheader()
        writer.writerows(test_data)
    
    # Test basic spreadsheet loading
    workflow = {
        "nodes": {
            "spreadsheet_loader": {
                "type": "LoadSpreadsheet",
                "config": {
                    "file_path": str(csv_path),
                    "text_column": "text"
                    # metadata_columns not specified = use all other columns
                }
            }
        },
        "connections": []
    }
    
    engine = WorkflowEngine()
    engine.load_workflow_from_dict(workflow)
    results = engine.execute()
    
    # Verify results
    assert "spreadsheet_loader" in results
    documents = results["spreadsheet_loader"]["documents"]
    assert len(documents) == 3
    
    # Check first document
    doc = documents[0]
    assert doc.page_content == "First document text"
    assert doc.metadata["id"] == 1
    assert doc.metadata["category"] == "A"
    assert doc.metadata["priority"] == "high"
    assert doc.metadata["row_number"] == 1
    assert doc.metadata["text_column"] == "text"
    assert doc.metadata["source"] == str(csv_path)
    
    print("✓ LoadSpreadsheet basic functionality")
    
    # Test with custom metadata columns
    workflow["nodes"]["spreadsheet_loader"]["config"]["metadata_columns"] = ["id", "category"]
    
    engine = WorkflowEngine()
    engine.load_workflow_from_dict(workflow)
    results = engine.execute()
    
    doc = results["spreadsheet_loader"]["documents"][0]
    assert "id" in doc.metadata
    assert "category" in doc.metadata
    assert "priority" not in doc.metadata  # Should be excluded
    print("✓ LoadSpreadsheet custom metadata columns")

def test_textsplitter_nodes(sample_doc_file):
    """Test all text splitter node types."""
    from onprem.workflow import WorkflowEngine
    
    # Test SplitByCharacterCount
    workflow = {
        "nodes": {
            "loader": {
                "type": "LoadSingleDocument",
                "config": {"file_path": sample_doc_file}
            },
            "splitter": {
                "type": "SplitByCharacterCount",
                "config": {"chunk_size": 20, "chunk_overlap": 5}
            }
        },
        "connections": [
            {
                "from": "loader", "from_port": "documents",
                "to": "splitter", "to_port": "documents"
            }
        ]
    }
    
    engine = WorkflowEngine()
    engine.load_workflow_from_dict(workflow)
    results = engine.execute()
    assert len(results["splitter"]["documents"]) >= 1
    print("✓ SplitByCharacterCount")
    
    # Test KeepFullDocument
    workflow["nodes"]["splitter"]["type"] = "KeepFullDocument" 
    workflow["nodes"]["splitter"]["config"] = {}
    
    engine = WorkflowEngine()
    engine.load_workflow_from_dict(workflow)
    results = engine.execute()
    assert len(results["splitter"]["documents"]) >= 1
    print("✓ KeepFullDocument")


def test_document_truncation(sample_doc_file):
    """Test document truncation functionality."""
    from onprem.workflow import WorkflowEngine
    
    # Create a longer test document for truncation (100 words)
    long_text = " ".join([f"word{i}" for i in range(100)])
    Path(sample_doc_file).write_text(long_text)
    
    # Test truncation to 50 words
    workflow = {
        "nodes": {
            "loader": {
                "type": "LoadSingleDocument",
                "config": {"file_path": sample_doc_file}
            },
            "truncate": {
                "type": "KeepFullDocument", 
                "config": {"max_words": 50}
            }
        },
        "connections": [
            {"from": "loader", "from_port": "documents", "to": "truncate", "to_port": "documents"}
        ]
    }
    
    engine = WorkflowEngine()
    engine.load_workflow_from_dict(workflow)
    results = engine.execute(verbose=False)
    
    truncated_doc = results["truncate"]["documents"][0]
    word_count = len(truncated_doc.page_content.split())
    
    # Verify truncation worked
    assert word_count == 50, f"Expected 50 words, got {word_count}"
    assert truncated_doc.metadata.get("truncated") == True
    assert truncated_doc.metadata.get("original_word_count") == 100
    assert truncated_doc.metadata.get("truncated_word_count") == 50
    print("✓ Document truncation (50 words)")
    
    # Test no truncation when document is shorter than limit
    workflow["nodes"]["truncate"]["config"]["max_words"] = 200
    engine.load_workflow_from_dict(workflow)
    results = engine.execute(verbose=False)
    
    no_truncate_doc = results["truncate"]["documents"][0]
    no_truncate_word_count = len(no_truncate_doc.page_content.split())
    
    # Should keep all words when under limit
    assert no_truncate_word_count == 100, f"Expected 100 words, got {no_truncate_word_count}"
    assert not no_truncate_doc.metadata.get("truncated", False)
    print("✓ No truncation when under limit")


def test_storage_nodes(temp_dir, sample_doc_file):
    """Test storage node types."""
    from onprem.workflow import WorkflowEngine
    
    # Test WhooshStore (need splitter between loader and storage)
    whoosh_path = os.path.join(temp_dir, "test_whoosh")
    workflow = {
        "nodes": {
            "loader": {
                "type": "LoadSingleDocument",
                "config": {"file_path": sample_doc_file}
            },
            "splitter": {
                "type": "KeepFullDocument",
                "config": {}
            },
            "storage": {
                "type": "WhooshStore",
                "config": {"persist_location": whoosh_path}
            }
        },
        "connections": [
            {
                "from": "loader", "from_port": "documents",
                "to": "splitter", "to_port": "documents"
            },
            {
                "from": "splitter", "from_port": "documents",
                "to": "storage", "to_port": "documents"
            }
        ]
    }
    
    engine = WorkflowEngine()
    engine.load_workflow_from_dict(workflow)
    results = engine.execute()
    assert "Successfully stored" in results["storage"]["status"]
    print("✓ WhooshStore")

def test_query_nodes(temp_dir, sample_doc_file):
    """Test query node types."""
    from onprem.workflow import WorkflowEngine
    
    # First create a Whoosh store
    whoosh_path = os.path.join(temp_dir, "test_whoosh_query")
    store_workflow = {
        "nodes": {
            "loader": {
                "type": "LoadSingleDocument", 
                "config": {"file_path": sample_doc_file}
            },
            "splitter": {
                "type": "KeepFullDocument",
                "config": {}
            },
            "storage": {
                "type": "WhooshStore",
                "config": {"persist_location": whoosh_path}
            }
        },
        "connections": [
            {
                "from": "loader", "from_port": "documents",
                "to": "splitter", "to_port": "documents"
            },
            {
                "from": "splitter", "from_port": "documents",
                "to": "storage", "to_port": "documents" 
            }
        ]
    }
    
    engine = WorkflowEngine()
    engine.load_workflow_from_dict(store_workflow)
    engine.execute()
    
    # Now test QueryWhooshStore
    query_workflow = {
        "nodes": {
            "query": {
                "type": "QueryWhooshStore",
                "config": {
                    "persist_location": whoosh_path,
                    "query": "test",
                    "limit": 5
                }
            }
        },
        "connections": []
    }
    
    engine = WorkflowEngine()
    engine.load_workflow_from_dict(query_workflow)
    results = engine.execute()
    assert "query" in results
    print("✓ QueryWhooshStore")

def test_exporter_nodes(temp_dir, sample_doc_file):
    """Test exporter node types."""
    from onprem.workflow import WorkflowEngine
    
    # Create mock results data
    mock_results = [
        {"content": "Test result 1", "score": 0.95},
        {"content": "Test result 2", "score": 0.85}
    ]
    
    # Test CSVExporter  
    csv_path = os.path.join(temp_dir, "test_results.csv")
    workflow = {
        "nodes": {
            "exporter": {
                "type": "CSVExporter",
                "config": {"output_path": csv_path}
            }
        },
        "connections": []
    }
    
    engine = WorkflowEngine()
    engine.load_workflow_from_dict(workflow)
    # Manually test exporter node directly
    exporter_node = engine.nodes["exporter"]
    results = exporter_node.execute({"results": mock_results})
    
    print(f"CSV Exporter result: {results}")
    assert "status" in results
    assert os.path.exists(csv_path)
    print("✓ CSVExporter")
    
    # Test JSONExporter
    json_path = os.path.join(temp_dir, "test_results.json") 
    workflow["nodes"]["exporter"]["type"] = "JSONExporter"
    workflow["nodes"]["exporter"]["config"]["output_path"] = json_path
    
    engine = WorkflowEngine()
    engine.load_workflow_from_dict(workflow)
    results = engine.nodes["exporter"].execute({"results": mock_results})
    
    print(f"JSON Exporter result: {results}")
    assert "status" in results
    assert os.path.exists(json_path)
    print("✓ JSONExporter")
    
    # Test JSONResponseExporter
    json_response_path = os.path.join(temp_dir, "test_json_responses.json")
    workflow["nodes"]["exporter"]["type"] = "JSONResponseExporter"
    workflow["nodes"]["exporter"]["config"]["output_path"] = json_response_path
    
    # Mock results with JSON-like responses
    mock_json_results = [
        {"response": '{"name": "John Doe", "skills": ["Python", "AI"], "experience": 5}'},
        {"response": '{"name": "Jane Smith", "skills": ["Java", "ML"], "experience": 3}'},
        {"output": '{"company": "TechCorp", "position": "Engineer", "rating": 4.5}'}
    ]
    
    engine = WorkflowEngine()
    engine.load_workflow_from_dict(workflow)
    
    # Mock the extract_json function
    from unittest.mock import patch
    with patch('onprem.llm.helpers.extract_json') as mock_extract:
        def mock_extract_side_effect(text):
            # Simple JSON extraction for test
            import json
            try:
                return json.loads(text)
            except:
                return None
        
        mock_extract.side_effect = mock_extract_side_effect
        
        results = engine.nodes["exporter"].execute({"results": mock_json_results})
        
        print(f"JSON Response Exporter result: {results}")
        assert "status" in results
        assert os.path.exists(json_response_path)
        
        # Verify the exported content
        with open(json_response_path, 'r') as f:
            import json
            exported_data = json.load(f)
            assert len(exported_data) == 3
            assert exported_data[0]["name"] == "John Doe"
            assert exported_data[1]["name"] == "Jane Smith"
            assert exported_data[2]["company"] == "TechCorp"
        
        print("✓ JSONResponseExporter")

def test_processor_nodes(temp_dir, sample_doc_file):
    """Test processor node types with mock LLM."""
    from onprem.workflow import WorkflowEngine
    from unittest.mock import patch, MagicMock
    
    # Mock LLM response
    mock_llm = MagicMock()
    mock_llm.predict.return_value = "Mock analysis: This is a test document about testing."
    
    with patch('onprem.llm.base.LLM') as mock_llm_class:
        mock_llm_class.return_value = mock_llm
        
        # Test PromptProcessor (DocumentProcessor)
        workflow = {
            "nodes": {
                "loader": {
                    "type": "LoadSingleDocument",
                    "config": {"file_path": sample_doc_file}
                },
                "processor": {
                    "type": "PromptProcessor",
                    "config": {
                        "prompt": "Analyze this document: {content}",
                        "llm": {"model_url": "test://mock"}
                    }
                }
            },
            "connections": [
                {
                    "from": "loader", "from_port": "documents",
                    "to": "processor", "to_port": "documents"
                }
            ]
        }
        
        engine = WorkflowEngine()
        engine.load_workflow_from_dict(workflow)
        results = engine.execute()
        
        assert "processor" in results
        assert len(results["processor"]["results"]) > 0
        assert "response" in results["processor"]["results"][0]
        # Verify mock LLM was called
        mock_llm.prompt.assert_called()
        print("✓ PromptProcessor (DocumentProcessor)")
        
        # Test ResponseCleaner (ResultProcessor)  
        mock_results = [{"response": "Raw response with extra text."}]
        
        cleaner_workflow = {
            "nodes": {
                "cleaner": {
                    "type": "ResponseCleaner",
                    "config": {
                        "cleanup_prompt": "Clean this response: {response}",
                        "llm": {"model_url": "test://mock"}
                    }
                }
            },
            "connections": []
        }
        
        engine = WorkflowEngine()
        engine.load_workflow_from_dict(cleaner_workflow)
        cleaner_node = engine.nodes["cleaner"]
        results = cleaner_node.execute({"results": mock_results})
        
        assert "results" in results
        assert len(results["results"]) > 0
        print("✓ ResponseCleaner (ResultProcessor)")
        
        # Test SummaryProcessor (DocumentProcessor)
        summary_workflow = {
            "nodes": {
                "loader": {
                    "type": "LoadSingleDocument",
                    "config": {"file_path": sample_doc_file}
                },
                "summarizer": {
                    "type": "SummaryProcessor", 
                    "config": {
                        "llm": {"model_url": "test://mock"}
                    }
                }
            },
            "connections": [
                {
                    "from": "loader", "from_port": "documents",
                    "to": "summarizer", "to_port": "documents"
                }
            ]
        }
        
        engine = WorkflowEngine()
        engine.load_workflow_from_dict(summary_workflow)
        results = engine.execute()
        
        assert "summarizer" in results
        assert len(results["summarizer"]["results"]) > 0
        print("✓ SummaryProcessor (DocumentProcessor)")

def test_processor_inheritance():
    """Test processor node inheritance hierarchy."""
    from onprem.workflow import NODE_REGISTRY, DocumentProcessor, ResultProcessor
    
    # Test PromptProcessor is DocumentProcessor
    prompt_node = NODE_REGISTRY["PromptProcessor"]("test", {})
    assert isinstance(prompt_node, DocumentProcessor)
    assert not isinstance(prompt_node, ResultProcessor)
    print("✓ PromptProcessor inherits from DocumentProcessor")
    
    # Test ResponseCleaner is ResultProcessor  
    cleaner_node = NODE_REGISTRY["ResponseCleaner"]("test", {})
    assert isinstance(cleaner_node, ResultProcessor)
    assert not isinstance(cleaner_node, DocumentProcessor)
    print("✓ ResponseCleaner inherits from ResultProcessor")
    
    # Test SummaryProcessor is DocumentProcessor
    summary_node = NODE_REGISTRY["SummaryProcessor"]("test", {})
    assert isinstance(summary_node, DocumentProcessor)
    assert not isinstance(summary_node, ResultProcessor)
    print("✓ SummaryProcessor inherits from DocumentProcessor")


def test_query_node_registry():
    """Test that all query nodes are properly registered."""
    from onprem.workflow import NODE_REGISTRY, QueryNode
    
    # Test QueryWhooshStore
    whoosh_node = NODE_REGISTRY["QueryWhooshStore"]("test", {
        "persist_location": "/tmp/test",
        "query": "test"
    })
    assert isinstance(whoosh_node, QueryNode)
    print("✓ QueryWhooshStore properly registered")
    
    # Test QueryChromaStore  
    chroma_node = NODE_REGISTRY["QueryChromaStore"]("test", {
        "persist_location": "/tmp/test",
        "query": "test" 
    })
    assert isinstance(chroma_node, QueryNode)
    print("✓ QueryChromaStore properly registered")
    
    # Test QueryElasticsearchStore
    es_node = NODE_REGISTRY["QueryElasticsearchStore"]("test", {
        "persist_location": "http://localhost:9200",
        "index_name": "test_index",
        "query": "test"
    })
    assert isinstance(es_node, QueryNode)
    print("✓ QueryElasticsearchStore properly registered")


def test_query_search_types():
    """Test search_type parameter validation for all query nodes."""
    from onprem.workflow import NODE_REGISTRY
    
    # Test WhooshStore search types
    WhooshNode = NODE_REGISTRY["QueryWhooshStore"]
    
    # Valid search types for WhooshStore
    for search_type in ["sparse", "semantic"]:
        node = WhooshNode("test", {
            "persist_location": "/tmp/test", 
            "query": "test",
            "search_type": search_type
        })
        assert node.config["search_type"] == search_type
    print("✓ WhooshStore supports sparse and semantic search types")
    
    # Invalid search type for WhooshStore
    try:
        node = WhooshNode("test", {
            "persist_location": "/tmp/test",
            "query": "test", 
            "search_type": "hybrid"
        })
        node.execute({})
        assert False, "Should raise error for unsupported search_type"
    except Exception as e:
        assert "does not support hybrid search" in str(e)
    print("✓ WhooshStore correctly rejects hybrid search type")
    
    # Test ChromaStore search types
    ChromaNode = NODE_REGISTRY["QueryChromaStore"]
    
    # Valid search type for ChromaStore (only semantic)
    node = ChromaNode("test", {
        "persist_location": "/tmp/test",
        "query": "test",
        "search_type": "semantic"
    })
    assert node.config["search_type"] == "semantic"
    print("✓ ChromaStore supports semantic search type")
    
    # Invalid search type for ChromaStore
    try:
        node = ChromaNode("test", {
            "persist_location": "/tmp/test",
            "query": "test",
            "search_type": "sparse"
        })
        node.execute({})
        assert False, "Should raise error for unsupported search_type"
    except Exception as e:
        assert "does not support sparse search" in str(e)
    print("✓ ChromaStore correctly rejects sparse search type")
    
    # Test ElasticsearchStore search types
    ElasticsearchNode = NODE_REGISTRY["QueryElasticsearchStore"]
    
    # Valid search types for ElasticsearchStore
    for search_type in ["sparse", "semantic", "hybrid"]:
        node = ElasticsearchNode("test", {
            "persist_location": "http://localhost:9200",
            "index_name": "test",
            "query": "test",
            "search_type": search_type
        })
        assert node.config["search_type"] == search_type
    print("✓ ElasticsearchStore supports sparse, semantic, and hybrid search types")


def test_query_dual_store_node():
    """Test QueryDualStore node functionality."""
    from onprem.workflow import NODE_REGISTRY
    
    # Test QueryDualStore registration
    QueryDualStoreNode = NODE_REGISTRY["QueryDualStore"]
    
    # Test all search types
    for search_type in ["sparse", "semantic", "hybrid"]:
        node = QueryDualStoreNode("test", {
            "persist_location": "/tmp/test_dual",
            "query": "test query",
            "search_type": search_type,
            "limit": 5
        })
        assert node.config["search_type"] == search_type
        assert node.config["limit"] == 5
    print("✓ QueryDualStore supports sparse, semantic, and hybrid search types")
    
    # Test hybrid search with weights
    node = QueryDualStoreNode("test", {
        "persist_location": "/tmp/test_dual",
        "query": "test query",
        "search_type": "hybrid",
        "weights": [0.7, 0.3]
    })
    assert node.config["weights"] == [0.7, 0.3]
    print("✓ QueryDualStore supports custom weights for hybrid search")
    
    # Test input/output types
    assert node.get_input_types() == {}
    assert node.get_output_types() == {"documents": "List[Document]"}
    print("✓ QueryDualStore has correct input/output types")
    
    # Test that it's properly registered as a QueryNode
    from onprem.workflow.base import QueryNode
    assert isinstance(node, QueryNode)
    print("✓ QueryDualStore inherits from QueryNode")


def test_document_to_results_converter():
    """Test DocumentToResults converter node functionality."""
    from onprem.workflow import NODE_REGISTRY
    from langchain_core.documents import Document
    
    # Test DocumentToResults registration
    DocumentToResultsNode = NODE_REGISTRY["DocumentToResults"]
    
    # Create test documents
    test_docs = [
        Document(
            page_content="This is a test document about AI.",
            metadata={"source": "test1.txt", "category": "technology", "page": 1}
        ),
        Document(
            page_content="Second document discussing machine learning.",
            metadata={"source": "test2.pdf", "author": "John Doe", "page_count": 5}
        )
    ]
    
    # Test basic conversion
    node = DocumentToResultsNode("test", {})
    results = node.execute({"documents": test_docs})
    
    assert "results" in results
    assert len(results["results"]) == 2
    
    # Check first result
    result1 = results["results"][0]
    assert result1["document_id"] == 0
    assert result1["source"] == "test1.txt"
    assert result1["content"] == "This is a test document about AI."
    assert result1["content_length"] == len("This is a test document about AI.")
    assert result1["meta_category"] == "technology"
    assert result1["meta_page"] == 1
    print("✓ DocumentToResults basic conversion")
    
    # Test custom configuration
    custom_node = DocumentToResultsNode("test", {
        "include_content": False,
        "metadata_prefix": "doc_",
        "custom_fields": {"processed_by": "workflow", "version": "1.0"}
    })
    
    custom_results = custom_node.execute({"documents": test_docs})
    custom_result = custom_results["results"][0]
    
    assert "content" not in custom_result  # Content excluded
    assert custom_result["doc_category"] == "technology"  # Custom prefix
    assert custom_result["processed_by"] == "workflow"  # Custom field
    assert custom_result["version"] == "1.0"  # Custom field
    print("✓ DocumentToResults custom configuration")
    
    # Test input/output types
    assert node.get_input_types() == {"documents": "List[Document]"}
    assert node.get_output_types() == {"results": "List[Dict]"}
    print("✓ DocumentToResults has correct input/output types")
    
    # Test that it's properly registered as a DocumentTransformerNode
    from onprem.workflow.base import DocumentTransformerNode
    assert isinstance(node, DocumentTransformerNode)
    print("✓ DocumentToResults inherits from DocumentTransformerNode")
    
    # Test empty input handling
    empty_results = node.execute({"documents": []})
    assert empty_results["results"] == []
    print("✓ DocumentToResults handles empty input")


def test_python_processors():
    """Test custom Python code processor nodes."""
    from onprem.workflow import NODE_REGISTRY
    from langchain_core.documents import Document
    
    # Test PythonDocumentProcessor
    PythonDocNode = NODE_REGISTRY["PythonDocumentProcessor"]
    
    # Test basic document processing
    node = PythonDocNode("test", {
        "code": """
# Simple document analysis
word_count = len(content.split())
char_count = len(content)
first_word = content.split()[0] if content else ''

result['analysis'] = {
    'word_count': word_count,
    'char_count': char_count, 
    'first_word': first_word,
    'has_test': 'test' in content.lower()
}
result['processor'] = 'python_document'
"""
    })
    
    # Create test document
    test_doc = Document(
        page_content="This is a test document for Python processing",
        metadata={"source": "test.txt", "type": "sample"}
    )
    
    results = node.execute({"documents": [test_doc]})
    assert "results" in results
    assert len(results["results"]) == 1
    
    result = results["results"][0]
    assert result["analysis"]["word_count"] == 8
    assert result["analysis"]["first_word"] == "This"
    assert result["analysis"]["has_test"] == True
    assert result["processor"] == "python_document"
    assert result["source"] == "test.txt"
    print("✓ PythonDocumentProcessor basic functionality")
    
    # Test error handling
    error_node = PythonDocNode("test", {
        "code": "invalid_variable_that_does_not_exist"
    })
    
    try:
        error_node.execute({"documents": [test_doc]})
        assert False, "Should raise error for invalid code"
    except Exception as e:
        assert "Error executing Python code" in str(e)
    print("✓ PythonDocumentProcessor error handling")
    
    # Test PythonResultProcessor
    PythonResNode = NODE_REGISTRY["PythonResultProcessor"]
    
    # Test basic result processing
    node = PythonResNode("test", {
        "code": """
# Enhance analysis results
original_analysis = result.get('analysis', {})
word_count = original_analysis.get('word_count', 0)

processed_result['enhanced_analysis'] = {
    'original_word_count': word_count,
    'doubled_word_count': word_count * 2,
    'category': 'short' if word_count < 10 else 'long',
    'processing_time': '2023-01-01'
}
processed_result['processor_chain'] = ['document', 'result']
processed_result['status'] = 'enhanced'
"""
    })
    
    # Use result from document processor test
    test_results = [{
        "analysis": {"word_count": 8, "first_word": "This"},
        "processor": "python_document",
        "source": "test.txt"
    }]
    
    results = node.execute({"results": test_results})
    assert "results" in results
    assert len(results["results"]) == 1
    
    result = results["results"][0]
    assert result["enhanced_analysis"]["original_word_count"] == 8
    assert result["enhanced_analysis"]["doubled_word_count"] == 16
    assert result["enhanced_analysis"]["category"] == "short"
    assert result["status"] == "enhanced"
    print("✓ PythonResultProcessor basic functionality")
    
    # Test missing code configuration
    try:
        invalid_node = PythonDocNode("test", {})  # No code provided
        invalid_node.execute({"documents": [test_doc]})
        assert False, "Should raise error for missing code"
    except Exception as e:
        assert "Either 'code' or 'code_file' is required" in str(e)
    print("✓ Python processors require code configuration")
    
    # Test safe execution environment
    safe_node = PythonDocNode("test", {
        "code": """
# Test available safe operations (modules are pre-imported)
result['regex_test'] = bool(re.search(r'test', content))
result['json_test'] = json.dumps({'key': 'value'})
result['math_test'] = math.sqrt(16)
result['builtin_test'] = len(content.split())
"""
    })
    
    results = safe_node.execute({"documents": [test_doc]})
    result = results["results"][0]
    assert result["regex_test"] == True
    assert '"key": "value"' in result["json_test"]
    assert result["math_test"] == 4.0
    assert result["builtin_test"] == 8
    print("✓ Python processors safe execution environment")


def test_aggregator_nodes():
    """Test aggregator node types."""
    from onprem.workflow import NODE_REGISTRY, AggregatorProcessor
    from unittest.mock import patch, MagicMock
    
    # Mock LLM response
    mock_llm = MagicMock()
    mock_llm.prompt.return_value = "Top topics: technology, AI, automation (appearing in 80% of documents)"
    
    with patch('onprem.llm.base.LLM') as mock_llm_class:
        mock_llm_class.return_value = mock_llm
        
        # Test AggregatorNode (LLM-based aggregation)
        AggregatorNodeClass = NODE_REGISTRY["AggregatorNode"]
        
        # Test basic aggregation
        node = AggregatorNodeClass("test", {
            "prompt": "Analyze these {num_results} responses and identify the top topics:\n{responses}",
            "llm": {"model_url": "test://mock"}
        })
        
        # Create test results (topic keywords per document)
        test_results = [
            {"response": "technology, AI, machine learning"},
            {"response": "automation, robotics, AI"},
            {"response": "data science, AI, technology"},
            {"response": "machine learning, automation"}
        ]
        
        results = node.execute({"results": test_results})
        assert "result" in results
        result = results["result"]
        assert "aggregated_response" in result
        assert result["source_count"] == 4
        assert result["aggregation_method"] == "llm_prompt"
        assert "original_results" in result
        
        # Verify mock LLM was called with correct prompt structure
        mock_llm.prompt.assert_called()
        call_args = mock_llm.prompt.call_args[0][0]
        assert "Analyze these 4 responses" in call_args
        assert "Response 1: technology, AI, machine learning" in call_args
        print("✓ AggregatorNode basic functionality")
        
        # Test with different result formats
        diverse_results = [
            {"summary": "Document discusses AI trends"},
            {"output": "Key finding: automation benefits"},
            {"response": "Main topic: robotics"}, 
            {"custom_field": "value", "other": "data"}  # Will use str() conversion
        ]
        
        results = node.execute({"results": diverse_results})
        result = results["result"]
        assert result["source_count"] == 4
        print("✓ AggregatorNode handles diverse result formats")
        
        # Test empty results
        results = node.execute({"results": []})
        assert results["result"] == {}
        print("✓ AggregatorNode handles empty results")
        
        # Test inheritance
        assert isinstance(node, AggregatorProcessor)
        assert node.get_input_types() == {"results": "List[Dict]"}
        assert node.get_output_types() == {"result": "Dict"}
        print("✓ AggregatorNode correct inheritance and types")


def test_python_aggregator_node():
    """Test PythonAggregatorNode functionality."""
    from onprem.workflow import NODE_REGISTRY, AggregatorProcessor
    
    PythonAggregatorClass = NODE_REGISTRY["PythonAggregatorNode"]
    
    # Test topic frequency aggregation
    node = PythonAggregatorClass("test", {
        "code": """
# Count topic frequency across all responses
topic_counts = {}
for res in results:
    response = res.get('response', '')
    topics = [t.strip() for t in response.split(',') if t.strip()]
    for topic in topics:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

# Get top 3 most frequent topics
sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
top_topics = sorted_topics[:3]

result['top_topics'] = [{'topic': topic, 'frequency': count} for topic, count in top_topics]
result['total_unique_topics'] = len(topic_counts)
result['aggregation_type'] = 'frequency_analysis'
"""
    })
    
    # Create test results (topic keywords)
    test_results = [
        {"response": "AI, machine learning, technology"},
        {"response": "AI, automation, robotics"},  
        {"response": "technology, AI, data science"},
        {"response": "automation, machine learning"}
    ]
    
    results = node.execute({"results": test_results})
    assert "result" in results
    result = results["result"]
    
    # Verify aggregation worked correctly
    assert "top_topics" in result
    assert len(result["top_topics"]) == 3
    
    # AI should be most frequent (appears 3 times)
    top_topic = result["top_topics"][0]
    assert top_topic["topic"] == "AI"
    assert top_topic["frequency"] == 3
    
    assert result["total_unique_topics"] == 6
    assert result["aggregation_type"] == "frequency_analysis" 
    assert result["source_count"] == 4
    assert result["aggregation_method"] == "python_code"
    print("✓ PythonAggregatorNode topic frequency analysis")
    
    # Test summary aggregation
    summary_node = PythonAggregatorClass("test", {
        "code": """
# Aggregate summaries into key themes
all_summaries = []
total_length = 0

for res in results:
    summary = res.get('summary', res.get('response', ''))
    all_summaries.append(summary)
    total_length += len(summary)

# Simple aggregation stats
result['combined_summary'] = ' | '.join(all_summaries)
result['num_summaries'] = len(all_summaries)
result['avg_summary_length'] = total_length // len(all_summaries) if all_summaries else 0
result['themes'] = ['research', 'development', 'implementation']  # Mock themes
"""
    })
    
    summary_results = [
        {"summary": "Research shows promising results"},
        {"summary": "Development phase completed successfully"},
        {"summary": "Implementation requires careful planning"}
    ]
    
    results = summary_node.execute({"results": summary_results})
    result = results["result"]
    
    assert "combined_summary" in result
    assert result["num_summaries"] == 3
    assert result["avg_summary_length"] > 0
    assert len(result["themes"]) == 3
    print("✓ PythonAggregatorNode summary aggregation")
    
    # Test inheritance and types
    assert isinstance(summary_node, AggregatorProcessor)
    assert summary_node.get_input_types() == {"results": "List[Dict]"}
    assert summary_node.get_output_types() == {"result": "Dict"}
    print("✓ PythonAggregatorNode correct inheritance and types")
    
    # Test error handling
    error_node = PythonAggregatorClass("test", {
        "code": "undefined_variable_causes_error"
    })
    
    try:
        error_node.execute({"results": test_results})
        assert False, "Should raise error for invalid code"
    except Exception as e:
        assert "Error executing Python code" in str(e)
    print("✓ PythonAggregatorNode error handling")


def test_workflow_validation():
    """Test workflow validation functionality."""
    from onprem.workflow import WorkflowEngine
    
    # Test valid workflow
    valid_workflow = {
        "nodes": {
            "loader": {
                "type": "LoadFromFolder",
                "config": {"source_directory": "/tmp"}
            }
        },
        "connections": []
    }
    
    engine = WorkflowEngine()
    engine.load_workflow_from_dict(valid_workflow)
    # If we get here without exception, validation passed
    print("✓ Valid workflow validation")
    
    # Test invalid workflow (missing config)
    invalid_workflow = {
        "nodes": {
            "loader": {
                "type": "LoadFromFolder"
                # Missing required config
            }
        },
        "connections": []
    }
    
    try:
        engine = WorkflowEngine()
        engine.load_workflow_from_dict(invalid_workflow)
        assert False, "Should have failed validation"
    except Exception:
        print("✓ Invalid workflow validation")

def run_all_tests():
    """Run all workflow tests with cleanup."""
    print("🧪 Running quick workflow tests...")
    
    # Create temp directory for tests
    import tempfile
    temp_dir = tempfile.mkdtemp()
    sample_file = os.path.join(temp_dir, "sample.txt")
    
    try:
        Path(sample_file).write_text(TEST_TEXT_CONTENT)
        
        print("\n📁 Testing Loader Nodes:")
        test_loader_nodes(temp_dir, sample_file)
        
        print("\n📊 Testing Spreadsheet Loader:")
        test_spreadsheet_loader(temp_dir)
        
        print("\n✂️ Testing TextSplitter Nodes:")  
        test_textsplitter_nodes(sample_file)
        
        print("\n📏 Testing Document Truncation:")
        test_document_truncation(sample_file)
        
        print("\n🗄️ Testing Storage Nodes:")
        test_storage_nodes(temp_dir, sample_file)
        
        print("\n🔍 Testing Query Nodes:")
        test_query_nodes(temp_dir, sample_file)
        
        print("\n💾 Testing Exporter Nodes:")
        test_exporter_nodes(temp_dir, sample_file)
        
        print("\n🤖 Testing Processor Nodes:")
        test_processor_nodes(temp_dir, sample_file)
        
        print("\n🏗️ Testing Processor Inheritance:")
        test_processor_inheritance()
        
        print("\n🔍 Testing Query Node Registry:")
        test_query_node_registry()
        
        print("\n🔍 Testing Query Search Types:")
        test_query_search_types()
        
        print("\n🔄 Testing Query Dual Store Node:")
        test_query_dual_store_node()
        
        print("\n🔄 Testing Document To Results Converter:")
        test_document_to_results_converter()
        
        print("\n🐍 Testing Python Processors:")
        test_python_processors()
        
        print("\n🔄 Testing Aggregator Nodes:")
        test_aggregator_nodes()
        
        print("\n🐍 Testing Python Aggregator Node:")
        test_python_aggregator_node()
        
        print("\n✅ Testing Workflow Validation:")
        test_workflow_validation()
        
        print(f"\n🎉 All tests passed! Node types working correctly.")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("🧹 Cleanup completed.")

if __name__ == "__main__":
    run_all_tests()