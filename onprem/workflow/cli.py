"""
Command-line interface for the workflow engine.

This module provides a CLI for validating and executing YAML-based workflows.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any

from .convenience import load_workflow, execute_workflow
from .exceptions import WorkflowValidationError, NodeExecutionError


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m onprem.workflow",
        description="Execute YAML-based document processing workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m onprem.workflow workflow.yaml
  python -m onprem.workflow --validate workflow.yaml
  python -m onprem.workflow --quiet workflow.yaml
  python -m onprem.workflow --help

Workflow Structure:
  A workflow YAML file contains 'nodes' and 'connections' sections:
  
  nodes:
    my_loader:
      type: LoadFromFolder
      config:
        source_directory: "documents/"
    
    my_chunker:
      type: SplitByCharacterCount
      config:
        chunk_size: 500
        chunk_overlap: 50
  
  connections:
    - from: my_loader
      from_port: documents
      to: my_chunker
      to_port: documents

Node Types:
  Loaders:        LoadFromFolder, LoadSingleDocument, LoadWebDocument
  TextSplitters:  SplitByCharacterCount, SplitByParagraph, KeepFullDocument
  Storage:        ChromaStore, WhooshStore, ElasticsearchStore
  Query:          QueryWhooshStore (sparse, semantic), QueryChromaStore (semantic), 
                  QueryElasticsearchStore (sparse, semantic, hybrid)
  Processors:     PromptProcessor, ResponseCleaner, SummaryProcessor,
                  PythonDocumentProcessor, PythonResultProcessor
  Exporters:      CSVExporter, ExcelExporter, JSONExporter

For detailed documentation, see: nbs/tests/workflows/workflow_tutorial.md
        """
    )
    
    parser.add_argument(
        "workflow_file",
        nargs="?",
        help="Path to the YAML workflow file to execute"
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        default=True,
        help="Show detailed progress information (default: enabled)"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true", 
        help="Suppress progress output"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate workflow without executing it"
    )
    
    parser.add_argument(
        "--list-nodes",
        action="store_true",
        help="List all available node types"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="OnPrem Workflow Engine v1.0"
    )
    
    args = parser.parse_args()
    
    # Handle special commands that don't require workflow file
    if args.list_nodes:
        print("📋 Available Node Types:\n")
        
        categories = {
            "Loader Nodes": ["LoadFromFolder", "LoadSingleDocument", "LoadWebDocument"],
            "TextSplitter Nodes": ["SplitByCharacterCount", "SplitByParagraph", "KeepFullDocument"], 
            "Storage Nodes": ["ChromaStore", "WhooshStore", "ElasticsearchStore"],
            "Query Nodes": [
                "QueryWhooshStore (search_type: sparse, semantic)", 
                "QueryChromaStore (search_type: semantic)", 
                "QueryElasticsearchStore (search_type: sparse, semantic, hybrid)"
            ],
            "Processor Nodes": ["PromptProcessor", "ResponseCleaner", "SummaryProcessor", 
                               "PythonDocumentProcessor", "PythonResultProcessor"],
            "Exporter Nodes": ["CSVExporter", "ExcelExporter", "JSONExporter"]
        }
        
        for category, nodes in categories.items():
            print(f"  {category}:")
            for node in nodes:
                print(f"    - {node}")
            print()
        
        return 0
    
    # For other commands, workflow file is required
    if not args.workflow_file:
        parser.error("workflow_file is required unless using --list-nodes")
    
    # Check if workflow file exists
    workflow_path = Path(args.workflow_file)
    if not workflow_path.exists():
        print(f"❌ Workflow file not found: {args.workflow_file}")
        return 1
    
    # Determine verbosity
    verbose = args.verbose and not args.quiet
    
    # Validate or execute workflow
    try:
        if args.validate:
            print(f"🔍 Validating workflow: {args.workflow_file}")
            engine = load_workflow(str(workflow_path))
            print("✅ Workflow validation successful!")
            
            # Show workflow summary
            if verbose:
                node_count = len(engine.nodes)
                conn_count = len(engine.connections)
                print(f"📊 Workflow summary: {node_count} nodes, {conn_count} connections")
                
                # Group nodes by type
                node_types = {}
                for node in engine.nodes.values():
                    node_type = node.NODE_TYPE
                    if node_type not in node_types:
                        node_types[node_type] = 0
                    node_types[node_type] += 1
                
                print("📋 Node breakdown:")
                for node_type, count in sorted(node_types.items()):
                    print(f"  - {node_type}: {count}")
            
            return 0
        else:
            print(f"🚀 Executing workflow: {args.workflow_file}")
            results = execute_workflow(str(workflow_path), verbose=verbose)
            print(f"\n✅ Workflow completed successfully!")
            
            if verbose:
                # Show execution summary
                total_docs = 0
                exports = []
                
                for node_id, result in results.items():
                    if isinstance(result, dict):
                        if "documents" in result and isinstance(result["documents"], list):
                            total_docs += len(result["documents"])
                        elif "status" in result and ("exported" in result["status"].lower() or "saved" in result["status"].lower()):
                            exports.append(result["status"])
                
                if total_docs > 0:
                    print(f"📄 Total documents processed: {total_docs}")
                
                if exports:
                    print("💾 Export results:")
                    for export in exports:
                        print(f"  - {export}")
            
            return 0
            
    except WorkflowValidationError as e:
        print(f"❌ Workflow validation failed: {e}")
        return 1
    except NodeExecutionError as e:
        print(f"❌ Workflow execution failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())