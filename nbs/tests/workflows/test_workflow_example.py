#!/usr/bin/env python3
"""
Simple example demonstrating the workflow module with WhooshStore.

This example:
1. Loads documents from the sample_data/sotu folder
2. Chunks them into 300-character pieces with 50-character overlap
3. Stores them in a Whoosh search index

Run this from the project root directory:
    python nbs/tests/test_workflow_example.py
"""

import os
import sys
import tempfile
import shutil

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from onprem.pipelines.workflow import execute_workflow, WorkflowEngine


def test_basic_workflow():
    """Test the basic workflow functionality."""
    print("Testing basic workflow with WhooshStore...")
    
    # Change to the nbs/tests directory so relative paths work
    original_dir = os.getcwd()
    test_dir = os.path.dirname(__file__)
    os.chdir(test_dir)
    
    try:
        # Execute the workflow
        workflow_path = "yaml_examples/example_workflow.yaml"
        results = execute_workflow(workflow_path, verbose=True)
        
        print(f"\nWorkflow completed successfully!")
        print(f"Results: {results}")
        
        # Verify the Whoosh index was created
        index_path = "test_whoosh_index"
        if os.path.exists(index_path):
            print(f"\n✓ Whoosh index created at: {index_path}")
            print(f"Index contents: {os.listdir(index_path)}")
        else:
            print(f"\n✗ Whoosh index not found at: {index_path}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Workflow failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        os.chdir(original_dir)
        # Clean up the test index
        index_path = os.path.join(test_dir, "test_whoosh_index")
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
            print(f"\nCleaned up test index: {index_path}")


def test_workflow_validation():
    """Test workflow validation features."""
    print("\nTesting workflow validation...")
    
    # Test invalid node type
    invalid_workflow = {
        "nodes": {
            "bad_node": {
                "type": "NonExistentNodeType",
                "config": {}
            }
        },
        "connections": []
    }
    
    engine = WorkflowEngine()
    try:
        engine.load_workflow_from_dict(invalid_workflow)
        print("✗ Should have failed on invalid node type")
        return False
    except Exception as e:
        print(f"✓ Correctly caught invalid node type: {str(e)}")
    
    # Test invalid connection
    invalid_connection_workflow = {
        "nodes": {
            "loader": {
                "type": "LoadFromFolder",
                "config": {"source_directory": "/tmp"}
            },
            "storage": {
                "type": "WhooshStore", 
                "config": {"persist_location": "/tmp/test"}
            }
        },
        "connections": [
            {
                "from": "loader",
                "from_port": "documents", 
                "to": "storage",
                "to_port": "documents"
            }
        ]
    }
    
    try:
        engine.load_workflow_from_dict(invalid_connection_workflow)
        print("✗ Should have failed on invalid connection (Loader -> Storage)")
        return False
    except Exception as e:
        print(f"✓ Correctly caught invalid connection: {str(e)}")
    
    return True


if __name__ == "__main__":
    print("=== Workflow Module Example ===\n")
    
    success = True
    
    # Test basic workflow
    success &= test_basic_workflow()
    
    # Test validation
    success &= test_workflow_validation()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)