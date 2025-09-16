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
        
        print("\n✅ Testing Workflow Validation:")
        test_workflow_validation()
        
        print(f"\n🎉 All tests passed! Node types working correctly.")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("🧹 Cleanup completed.")

if __name__ == "__main__":
    run_all_tests()