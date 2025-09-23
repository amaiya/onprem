#!/usr/bin/env python3
"""
Simplified workflow tests with full coverage - consolidates repetitive tests 
while maintaining all essential functionality validation.
"""
import os
import tempfile
import shutil
import pytest
from pathlib import Path

# Test constants
TEST_TEXT_CONTENT = "This is a test document for workflow testing."
TRUNCATION_TEXT = " ".join([f"word{i}" for i in range(100)])

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture  
def sample_files(temp_dir):
    """Create sample files for testing."""
    files = {}
    
    # Main test document
    files['main'] = Path(temp_dir) / "test_doc.txt"
    files['main'].write_text(TEST_TEXT_CONTENT)
    
    # Truncation test document (100 words)
    files['truncation'] = Path(temp_dir) / "truncation_test.txt" 
    files['truncation'].write_text(TRUNCATION_TEXT)
    
    # Spreadsheet test file
    files['spreadsheet'] = Path(temp_dir) / "test_data.csv"
    files['spreadsheet'].write_text("""name,description,category,priority
John,AI research findings,technology,high
Jane,Meeting notes summary,business,medium  
Bob,Code review comments,technology,low""")
    
    return files

def build_workflow(nodes, connections=None):
    """Helper to build workflow dictionaries."""
    if connections is None:
        # Auto-generate linear connections
        node_ids = list(nodes.keys())
        connections = []
        for i in range(len(node_ids) - 1):
            from_id, to_id = node_ids[i], node_ids[i + 1]
            from_port = "documents" if i == 0 else "documents"
            to_port = "documents" if "Query" in nodes[to_id]["type"] else "documents"
            connections.append({
                "from": from_id, "from_port": from_port,
                "to": to_id, "to_port": to_port
            })
    
    return {"nodes": nodes, "connections": connections}

def execute_workflow(workflow, verbose=False):
    """Helper to execute workflows with error handling."""
    from onprem.workflow import WorkflowEngine
    
    engine = WorkflowEngine()
    engine.load_workflow_from_dict(workflow)
    return engine.execute(verbose=verbose)

class TestLoaderNodes:
    """Test all document loading functionality."""
    
    def test_file_loaders(self, sample_files):
        """Test LoadFromFolder and LoadSingleDocument."""
        temp_dir = str(sample_files['main'].parent)
        
        # Test LoadFromFolder - verify it loads files
        workflow = build_workflow({
            "loader": {
                "type": "LoadFromFolder",
                "config": {
                    "source_directory": temp_dir,
                    "include_patterns": ["*.txt"]
                }
            }
        })
        
        results = execute_workflow(workflow)
        docs = results["loader"]["documents"]
        assert len(docs) >= 1, "Should load at least one document"
        
        # Find our main test document 
        test_doc = next((d for d in docs if d.page_content == TEST_TEXT_CONTENT), None)
        assert test_doc is not None, "Should find our test document"
        
        # Test LoadSingleDocument  
        workflow = build_workflow({
            "loader": {
                "type": "LoadSingleDocument", 
                "config": {"file_path": str(sample_files['main'])}
            }
        })
        
        results = execute_workflow(workflow)
        assert len(results["loader"]["documents"]) == 1
        assert results["loader"]["documents"][0].page_content == TEST_TEXT_CONTENT

    def test_spreadsheet_loader(self, sample_files):
        """Test LoadSpreadsheet with different configurations."""
        # Test with specific metadata columns
        workflow = build_workflow({
            "loader": {
                "type": "LoadSpreadsheet",
                "config": {
                    "file_path": str(sample_files['spreadsheet']),
                    "text_column": "description", 
                    "metadata_columns": ["name", "category"]
                }
            }
        })
        
        results = execute_workflow(workflow)
        docs = results["loader"]["documents"]
        assert len(docs) == 3
        assert "research" in docs[0].page_content
        assert docs[0].metadata["name"] == "John"
        assert docs[0].metadata["category"] == "technology"


class TestDocumentProcessing:
    """Test document transformation and processing."""
    
    def test_text_splitters(self, sample_files):
        """Test all text splitting strategies."""
        workflows = {
            "character_split": build_workflow({
                "loader": {"type": "LoadSingleDocument", "config": {"file_path": str(sample_files['main'])}},
                "splitter": {"type": "SplitByCharacterCount", "config": {"chunk_size": 20, "chunk_overlap": 5}}
            }),
            "keep_full": build_workflow({
                "loader": {"type": "LoadSingleDocument", "config": {"file_path": str(sample_files['main'])}},
                "splitter": {"type": "KeepFullDocument", "config": {}}
            }),
            "truncate": build_workflow({
                "loader": {"type": "LoadSingleDocument", "config": {"file_path": str(sample_files['truncation'])}},
                "splitter": {"type": "KeepFullDocument", "config": {"max_words": 50}}
            })
        }
        
        # Test character splitting
        results = execute_workflow(workflows["character_split"])
        docs = results["splitter"]["documents"]
        assert len(docs) > 1  # Should split into multiple chunks
        
        # Test keep full document
        results = execute_workflow(workflows["keep_full"])
        docs = results["splitter"]["documents"] 
        assert len(docs) == 1
        assert docs[0].page_content == TEST_TEXT_CONTENT
        
        # Test truncation
        results = execute_workflow(workflows["truncate"])
        doc = results["splitter"]["documents"][0]
        word_count = len(doc.page_content.split())
        assert word_count == 50
        assert doc.metadata.get("truncated") == True

    def test_document_transformers(self, sample_files):
        """Test document transformation nodes."""
        workflow = build_workflow({
            "loader": {"type": "LoadSingleDocument", "config": {"file_path": str(sample_files['main'])}},
            "add_meta": {
                "type": "AddMetadata",
                "config": {"metadata": {"category": "test", "priority": "high"}}
            },
            "prefix": {
                "type": "ContentPrefix", 
                "config": {"prefix": "[TEST]", "separator": " "}
            },
            "filter": {
                "type": "DocumentFilter",
                "config": {"content_contains": ["test"], "min_length": 10}
            }
        })
        
        results = execute_workflow(workflow)
        doc = results["filter"]["documents"][0]
        assert doc.metadata["category"] == "test"
        assert doc.page_content.startswith("[TEST] This is a test")
        assert len(results["filter"]["documents"]) == 1  # Should pass filter


class TestStorageAndQuery:
    """Test storage and querying functionality."""
    
    def test_whoosh_store_and_query(self, sample_files):
        """Test complete store → query workflow."""
        temp_dir = str(sample_files['main'].parent)
        whoosh_path = os.path.join(temp_dir, "test_whoosh")
        
        # Store documents
        store_workflow = build_workflow({
            "loader": {"type": "LoadSingleDocument", "config": {"file_path": str(sample_files['main'])}},
            "splitter": {"type": "KeepFullDocument", "config": {}},
            "storage": {"type": "WhooshStore", "config": {"persist_location": whoosh_path}}
        })
        
        results = execute_workflow(store_workflow)
        assert "Successfully stored" in results["storage"]["status"]
        
        # Query documents
        query_workflow = build_workflow({
            "query": {
                "type": "QueryWhooshStore",
                "config": {
                    "persist_location": whoosh_path,
                    "query": "test",
                    "limit": 5,
                    "search_type": "sparse"
                }
            }
        })
        
        results = execute_workflow(query_workflow)
        docs = results["query"]["documents"]
        assert len(docs) > 0, "Query should find the test document"
        assert "test" in docs[0].page_content.lower()

    def test_search_types(self, sample_files):
        """Test different search type validations."""
        from onprem.workflow import NODE_REGISTRY
        
        # Test node type validations
        test_cases = [
            ("QueryWhooshStore", ["sparse", "semantic"], ["hybrid"]),
            ("QueryChromaStore", ["semantic"], ["sparse", "hybrid"]),
            ("QueryElasticsearchStore", ["sparse", "semantic", "hybrid"], []),
            ("QueryDualStore", ["sparse", "semantic", "hybrid"], [])
        ]
        
        for node_type, valid_types, invalid_types in test_cases:
            node_class = NODE_REGISTRY[node_type]
            
            # Test valid types work (create instance without error)
            for search_type in valid_types:
                try:
                    node = node_class("test", {"search_type": search_type})
                    assert hasattr(node, "NODE_TYPE")
                except Exception as e:
                    pytest.fail(f"{node_type} should support {search_type}: {e}")
            
            # Test invalid types are rejected
            for search_type in invalid_types:
                node = node_class("test", {
                    "persist_location": "/tmp/test", 
                    "query": "test",
                    "search_type": search_type
                })
                try:
                    # This should raise an error during execution
                    node.execute({})
                    pytest.fail(f"{node_type} should reject {search_type}")
                except Exception:
                    pass  # Expected to fail


class TestProcessors:
    """Test document processing and analysis nodes."""
    
    def test_python_processors(self, sample_files):
        """Test Python code execution processors."""
        # Test PythonDocumentProcessor
        doc_workflow = build_workflow({
            "loader": {"type": "LoadSingleDocument", "config": {"file_path": str(sample_files['main'])}},
            "processor": {
                "type": "PythonDocumentProcessor",
                "config": {
                    "code": '''
# Simple analysis
word_count = len(content.split())
result['word_count'] = word_count
result['has_test'] = 'test' in content.lower()
result['source_file'] = source
'''
                }
            }
        })
        
        results = execute_workflow(doc_workflow)
        result = results["processor"]["results"][0]
        assert result["word_count"] > 0
        assert result["has_test"] == True
        assert "test_doc.txt" in result["source_file"]
        
        # Test PythonResultProcessor 
        result_workflow = build_workflow({
            "loader": {"type": "LoadSingleDocument", "config": {"file_path": str(sample_files['main'])}},
            "doc_proc": {
                "type": "PythonDocumentProcessor", 
                "config": {"code": "result['word_count'] = len(content.split())"}
            },
            "result_proc": {
                "type": "PythonResultProcessor",
                "config": {
                    "code": '''
# Enhance results
word_count = result.get('word_count', 0)
processed_result['category'] = 'long' if word_count > 5 else 'short'
processed_result['enhanced'] = True
'''
                }
            }
        }, connections=[
            {"from": "loader", "from_port": "documents", "to": "doc_proc", "to_port": "documents"},
            {"from": "doc_proc", "from_port": "results", "to": "result_proc", "to_port": "results"}
        ])
        
        results = execute_workflow(result_workflow)
        result = results["result_proc"]["results"][0]
        assert result["category"] in ["long", "short"]
        assert result["enhanced"] == True

    def test_aggregators(self, sample_files):
        """Test result aggregation functionality."""
        # Create multiple results to aggregate
        workflow = build_workflow({
            "loader": {"type": "LoadSingleDocument", "config": {"file_path": str(sample_files['main'])}},
            "processor": {
                "type": "PythonDocumentProcessor",
                "config": {"code": "result['topic'] = 'testing'; result['score'] = 95"}
            },
            "aggregator": {
                "type": "PythonAggregatorNode",
                "config": {
                    "code": '''
# Aggregate results
total_score = sum(r.get('score', 0) for r in results)
result['average_score'] = total_score / len(results) if results else 0
result['total_documents'] = len(results)
result['topics'] = [r.get('topic', 'unknown') for r in results]
'''
                }
            }
        }, connections=[
            {"from": "loader", "from_port": "documents", "to": "processor", "to_port": "documents"},
            {"from": "processor", "from_port": "results", "to": "aggregator", "to_port": "results"}
        ])
        
        results = execute_workflow(workflow) 
        agg_result = results["aggregator"]["result"]
        assert agg_result["average_score"] == 95
        assert agg_result["total_documents"] == 1
        assert agg_result["topics"] == ["testing"]


class TestExportAndIntegration:
    """Test export functionality and end-to-end workflows."""
    
    def test_exporters(self, sample_files, temp_dir):
        """Test all export formats."""
        # Create some results to export
        workflow = build_workflow({
            "loader": {"type": "LoadSingleDocument", "config": {"file_path": str(sample_files['main'])}},
            "converter": {"type": "DocumentToResults", "config": {"include_content": True}},
            "csv_export": {"type": "CSVExporter", "config": {"output_path": f"{temp_dir}/test.csv"}},
            "json_export": {"type": "JSONExporter", "config": {"output_path": f"{temp_dir}/test.json"}}
        }, connections=[
            {"from": "loader", "from_port": "documents", "to": "converter", "to_port": "documents"},
            {"from": "converter", "from_port": "results", "to": "csv_export", "to_port": "results"},
            {"from": "converter", "from_port": "results", "to": "json_export", "to_port": "results"}
        ])
        
        results = execute_workflow(workflow)
        
        # Verify exports succeeded
        assert "Exported" in results["csv_export"]["status"]
        assert "Exported" in results["json_export"]["status"]
        assert os.path.exists(f"{temp_dir}/test.csv")
        assert os.path.exists(f"{temp_dir}/test.json")

    def test_complete_workflow(self, sample_files):
        """Test complete end-to-end workflow with all major components."""
        temp_dir = str(sample_files['main'].parent)
        whoosh_path = os.path.join(temp_dir, "complete_test")
        csv_path = os.path.join(temp_dir, "complete_results.csv")
        
        # Complete workflow: Load → Store → Query → Process → Export
        complete_workflow = {
            "nodes": {
                # Phase 1: Ingestion
                "loader": {"type": "LoadSingleDocument", "config": {"file_path": str(sample_files['main'])}},
                "splitter": {"type": "KeepFullDocument", "config": {}},
                "storage": {"type": "WhooshStore", "config": {"persist_location": whoosh_path}},
                
                # Phase 2: Query and Process  
                "query": {
                    "type": "QueryWhooshStore",
                    "config": {"persist_location": whoosh_path, "query": "test", "limit": 10}
                },
                "processor": {
                    "type": "PythonDocumentProcessor",
                    "config": {"code": "result['analysis'] = 'complete'; result['found_test'] = 'test' in content"}
                },
                
                # Phase 3: Export
                "exporter": {"type": "CSVExporter", "config": {"output_path": csv_path}}
            },
            "connections": [
                # Ingestion pipeline
                {"from": "loader", "from_port": "documents", "to": "splitter", "to_port": "documents"},
                {"from": "splitter", "from_port": "documents", "to": "storage", "to_port": "documents"},
                
                # Query and processing pipeline  
                {"from": "query", "from_port": "documents", "to": "processor", "to_port": "documents"},
                {"from": "processor", "from_port": "results", "to": "exporter", "to_port": "results"}
            ]
        }
        
        # Execute ingestion phase
        from onprem.workflow import WorkflowEngine
        
        # First run ingestion
        ingestion_workflow = {
            "nodes": {k: v for k, v in complete_workflow["nodes"].items() if k in ["loader", "splitter", "storage"]},
            "connections": [c for c in complete_workflow["connections"] if c["to"] in ["splitter", "storage"]]
        }
        
        engine = WorkflowEngine()
        engine.load_workflow_from_dict(ingestion_workflow)
        engine.execute()
        
        # Then run processing 
        processing_workflow = {
            "nodes": {k: v for k, v in complete_workflow["nodes"].items() if k in ["query", "processor", "exporter"]},
            "connections": [c for c in complete_workflow["connections"] if c["from"] in ["query", "processor"]]
        }
        
        engine.load_workflow_from_dict(processing_workflow)
        results = engine.execute()
        
        # Verify complete pipeline worked
        assert "Exported" in results["exporter"]["status"]
        assert os.path.exists(csv_path)
        
        # Verify processing results
        proc_results = results["processor"]["results"]
        assert len(proc_results) > 0
        assert proc_results[0]["analysis"] == "complete"
        assert proc_results[0]["found_test"] == True


class TestValidation:
    """Test workflow validation and error handling."""
    
    def test_node_registry(self):
        """Test all expected nodes are registered."""
        from onprem.workflow import NODE_REGISTRY
        
        expected_nodes = [
            # Loaders
            "LoadFromFolder", "LoadSingleDocument", "LoadSpreadsheet",
            # Splitters  
            "SplitByCharacterCount", "KeepFullDocument",
            # Transformers
            "AddMetadata", "ContentPrefix", "ContentSuffix", "DocumentFilter", "PythonDocumentTransformer",
            # Storage
            "WhooshStore", "ChromaStore", "ElasticsearchStore", 
            # Query
            "QueryWhooshStore", "QueryChromaStore", "QueryElasticsearchStore", "QueryDualStore",
            # Processors
            "PromptProcessor", "ResponseCleaner", "SummaryProcessor", "PythonDocumentProcessor", 
            "PythonResultProcessor", "DocumentToResults", "AggregatorNode", "PythonAggregatorNode",
            # Exporters
            "CSVExporter", "JSONExporter", "JSONResponseExporter"
        ]
        
        for node_name in expected_nodes:
            assert node_name in NODE_REGISTRY, f"Node {node_name} not registered"
            
        # Test node inheritance
        from onprem.workflow.base import DocumentProcessor, ResultProcessor, AggregatorProcessor
        
        assert issubclass(NODE_REGISTRY["PromptProcessor"], DocumentProcessor)
        assert issubclass(NODE_REGISTRY["ResponseCleaner"], ResultProcessor)  
        assert issubclass(NODE_REGISTRY["AggregatorNode"], AggregatorProcessor)

    def test_workflow_validation(self):
        """Test workflow validation catches errors."""
        from onprem.workflow import WorkflowEngine
        
        # Test invalid node type
        invalid_workflow = {
            "nodes": {"invalid": {"type": "NonExistentNode", "config": {}}},
            "connections": []
        }
        
        engine = WorkflowEngine()
        with pytest.raises(Exception, match="Unknown node type"):
            engine.load_workflow_from_dict(invalid_workflow)


# Simplified test runner for backwards compatibility
def run_all_tests():
    """Run all tests - maintains compatibility with existing test runner."""
    import pytest
    
    print("🧪 Running simplified workflow tests...")
    
    # Run pytest on this module
    result = pytest.main([__file__, "-v", "--tb=short"])
    
    if result == 0:
        print("🎉 All tests passed! Workflow functionality verified.")
    else:
        print("❌ Some tests failed. Check output above.")
    
    return result == 0


if __name__ == "__main__":
    run_all_tests()