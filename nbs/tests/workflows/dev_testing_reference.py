#!/usr/bin/env python3
"""
Development reference for testing new workflow nodes with mock data.

This file demonstrates how to:
1. Test node type registration
2. Validate workflow connections  
3. Create mock processors without LLM dependencies
4. Test exporters with sample data

Useful for development and debugging new node types.
"""

import os
import sys
from langchain_core.documents import Document

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)

from onprem.pipelines.workflow import WorkflowEngine, NODE_REGISTRY


def test_new_node_types():
    """Test that new node types are registered correctly."""
    print("Testing node type registration...")
    
    expected_new_nodes = [
        "QueryWhooshStore", "QueryChromaStore",
        "PromptProcessor", "SummaryProcessor", 
        "CSVExporter", "ExcelExporter", "JSONExporter"
    ]
    
    for node_type in expected_new_nodes:
        if node_type in NODE_REGISTRY:
            print(f"✓ {node_type} registered")
        else:
            print(f"✗ {node_type} NOT registered")
    
    return True


def test_validation():
    """Test that the new validation system works with new node types."""
    print("\nTesting validation with new node types...")
    
    # Test valid workflow: Query -> Processor -> Exporter
    valid_workflow = {
        "nodes": {
            "query": {
                "type": "QueryWhooshStore",
                "config": {"persist_location": "/tmp", "query": "test"}
            },
            "processor": {
                "type": "PromptProcessor", 
                "config": {"prompt": "Test prompt: {content}"}
            },
            "exporter": {
                "type": "CSVExporter",
                "config": {"output_path": "test.csv"}
            }
        },
        "connections": [
            {"from": "query", "from_port": "documents", "to": "processor", "to_port": "documents"},
            {"from": "processor", "from_port": "results", "to": "exporter", "to_port": "results"}
        ]
    }
    
    engine = WorkflowEngine()
    try:
        engine.load_workflow_from_dict(valid_workflow)
        print("✓ Valid Query->Processor->Exporter workflow accepted")
    except Exception as e:
        print(f"✗ Valid workflow rejected: {e}")
        return False
    
    # Test invalid workflow: Storage -> Query (Storage is terminal)
    invalid_workflow = {
        "nodes": {
            "storage": {
                "type": "WhooshStore",
                "config": {"persist_location": "/tmp"}
            },
            "query": {
                "type": "QueryWhooshStore",
                "config": {"persist_location": "/tmp", "query": "test"}
            }
        },
        "connections": [
            {"from": "storage", "from_port": "status", "to": "query", "to_port": "documents"}
        ]
    }
    
    try:
        engine.load_workflow_from_dict(invalid_workflow)
        print("✗ Invalid Storage->Query workflow should have been rejected")
        return False
    except Exception as e:
        print(f"✓ Invalid workflow correctly rejected: {e}")
    
    return True


def create_mock_documents():
    """Create mock documents for testing."""
    return [
        Document(
            page_content="Artificial intelligence is revolutionizing healthcare through advanced diagnostics.",
            metadata={"source": "healthcare_ai.pdf", "page": 1}
        ),
        Document(
            page_content="Machine learning algorithms can predict patient outcomes with high accuracy.",
            metadata={"source": "ml_prediction.pdf", "page": 1}
        ),
        Document(
            page_content="Deep learning neural networks excel at image recognition tasks in medical imaging.",
            metadata={"source": "medical_imaging.pdf", "page": 2}
        )
    ]


def test_mock_processor():
    """Test processor nodes with mock data (without LLM)."""
    print("\nTesting processor with mock data...")
    
    try:
        from onprem.pipelines.workflow import PromptProcessorNode
        
        # Create a mock processor that doesn't use LLM
        class MockPromptProcessor(PromptProcessorNode):
            def execute(self, inputs):
                documents = inputs.get("documents", [])
                if not documents:
                    return {"results": []}
                
                prompt_template = self.config.get("prompt", "")
                results = []
                
                for i, doc in enumerate(documents):
                    # Mock LLM response
                    mock_response = f"TOPIC: Healthcare AI\nFINDINGS: Advanced diagnostics\nCOMPLEXITY: 4\nTECHNOLOGIES: AI, ML"
                    
                    result = {
                        'document_id': i,
                        'source': doc.metadata.get('source', 'Unknown'),
                        'prompt': prompt_template.format(content=doc.page_content, **doc.metadata),
                        'response': mock_response,
                        'metadata': doc.metadata
                    }
                    results.append(result)
                
                return {"results": results}
        
        # Test the mock processor
        processor = MockPromptProcessor("test_processor", {
            "prompt": "Analyze: {content}"
        })
        
        mock_docs = create_mock_documents()
        results = processor.execute({"documents": mock_docs})
        
        print(f"✓ Processed {len(results['results'])} documents")
        for result in results["results"][:1]:  # Show first result
            print(f"  - Source: {result['source']}")
            print(f"  - Response: {result['response'][:50]}...")
        
        return results["results"]
        
    except Exception as e:
        print(f"✗ Mock processor test failed: {e}")
        return []


def test_csv_exporter(results):
    """Test CSV exporter with mock results."""
    print("\nTesting CSV exporter...")
    
    try:
        from onprem.pipelines.workflow import CSVExporterNode
        
        exporter = CSVExporterNode("test_exporter", {
            "output_path": "test_results.csv"
        })
        
        export_result = exporter.execute({"results": results})
        print(f"✓ {export_result['status']}")
        
        # Verify file was created
        if os.path.exists("test_results.csv"):
            with open("test_results.csv", 'r') as f:
                lines = f.readlines()
                print(f"  - CSV file has {len(lines)} lines (including header)")
            
            # Clean up
            os.remove("test_results.csv")
            print("  - Cleaned up test file")
        
        return True
        
    except Exception as e:
        print(f"✗ CSV exporter test failed: {e}")
        return False


if __name__ == "__main__":
    print("=== Testing New Workflow Nodes ===\n")
    
    success = True
    
    # Test registration
    success &= test_new_node_types()
    
    # Test validation
    success &= test_validation()
    
    # Test mock processing
    results = test_mock_processor()
    if results:
        success &= test_csv_exporter(results)
    else:
        success = False
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)