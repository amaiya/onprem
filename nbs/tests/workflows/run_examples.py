#!/usr/bin/env python3
"""
Easy workflow example runner and tester.

Usage:
    python run_examples.py --list                    # List all examples
    python run_examples.py --run basic              # Run basic example
    python run_examples.py --run pattern_filter     # Run pattern filtering
    python run_examples.py --validate all           # Validate all workflows
    python run_examples.py --help                   # Show help
"""

import os
import sys
import argparse
import glob
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from onprem.pipelines.workflow import execute_workflow, load_workflow
    ONPREM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  OnPrem not available: {e}")
    ONPREM_AVAILABLE = False


# Example metadata - describes what each example does and requirements
EXAMPLES = {
    "example_workflow.yaml": {
        "name": "Basic Pipeline",
        "description": "Simple Load → Chunk → Store pipeline using WhooshStore",
        "requirements": ["sample_data/sotu directory"],
        "creates": ["test_whoosh_index/"],
        "difficulty": "Beginner"
    },
    "advanced_workflow_example.yaml": {
        "name": "Multi-Source Pipeline", 
        "description": "Multiple loaders with different chunking strategies",
        "requirements": ["sample_data/sotu", "sample_data/billionaires"],
        "creates": ["unified_search_index/"],
        "difficulty": "Intermediate"
    },
    "pattern_pdf_only.yaml": {
        "name": "Pattern Filtering - PDF Only",
        "description": "Load only PDF files using include_patterns filtering",
        "requirements": ["sample_data/ with PDF files"],
        "creates": ["pdf_documents/"],
        "difficulty": "Intermediate"
    },
    "query_and_prompt_example.yaml": {
        "name": "Query & Prompt Analysis",
        "description": "Query storage index → Apply AI prompt → Export to Excel",
        "requirements": ["existing_search_index/", "LLM API key"],
        "creates": ["document_analysis_results.xlsx"],
        "difficulty": "Advanced",
        "note": "Requires existing populated index and LLM"
    },
    "complete_analysis_pipeline.yaml": {
        "name": "Complete Analysis Pipeline",
        "description": "Full pipeline: Ingest → Store → Query → Process → Export",
        "requirements": ["sample_data/ with PDFs", "LLM API key"],
        "creates": ["analysis_index/", "research_summaries.csv"],
        "difficulty": "Advanced",
        "note": "Query step runs independently - see tutorial for explanation"
    },
    "simple_analysis_demo.yaml": {
        "name": "Simple Analysis Demo",
        "description": "Query existing index → Apply prompt → Export CSV",
        "requirements": ["existing_index/", "LLM API key"],
        "creates": ["ai_document_analysis.csv"],
        "difficulty": "Advanced",
        "note": "Requires pre-existing populated index"
    }
}


def list_examples():
    """List all available workflow examples."""
    print("📋 Available Workflow Examples:\n")
    
    for i, (filename, info) in enumerate(EXAMPLES.items(), 1):
        status = "✅" if os.path.exists(filename) else "❌"
        print(f"{i:2d}. {status} {info['name']} ({info['difficulty']})")
        print(f"     File: {filename}")
        print(f"     Description: {info['description']}")
        
        if info['requirements']:
            print(f"     Requirements: {', '.join(info['requirements'])}")
        
        if info.get('note'):
            print(f"     Note: {info['note']}")
        
        print()


def check_requirements(example_file):
    """Check if requirements for an example are met."""
    if example_file not in EXAMPLES:
        return True, []
    
    info = EXAMPLES[example_file]
    requirements = info.get('requirements', [])
    missing = []
    
    for req in requirements:
        if req.startswith('sample_data/'):
            # Check if sample data directory exists
            path = f"../{req}"
            if not os.path.exists(path):
                missing.append(f"{req} (not found at {path})")
        elif req.endswith('/'):
            # Directory requirement
            if not os.path.exists(req):
                missing.append(f"{req} (directory not found)")
        elif 'API key' in req:
            # API key requirement
            missing.append(f"{req} (check environment variables)")
        elif req.endswith('_index/') or 'existing' in req:
            # Existing index requirement
            if not os.path.exists(req.replace('existing_', '').replace('_index/', '_index')):
                missing.append(f"{req} (pre-existing index required)")
    
    return len(missing) == 0, missing


def validate_workflow(yaml_file):
    """Validate a workflow file without executing it."""
    if not ONPREM_AVAILABLE:
        return False, "OnPrem not available"
    
    try:
        workflow_engine = load_workflow(yaml_file)
        return True, "Validation successful"
    except Exception as e:
        return False, str(e)


def run_workflow(yaml_file, dry_run=False):
    """Run a workflow with error handling and cleanup."""
    if not ONPREM_AVAILABLE:
        print("❌ Cannot run workflow: OnPrem not available")
        return False
    
    # Check requirements
    req_ok, missing = check_requirements(yaml_file)
    if not req_ok:
        print("⚠️  Missing requirements:")
        for req in missing:
            print(f"   - {req}")
        print("\nYou may need to:")
        print("   - Ensure sample_data/ exists with documents")
        print("   - Set up LLM API keys (OPENAI_API_KEY, etc.)")  
        print("   - Run a storage workflow first to create indexes")
        return False
    
    if dry_run:
        print("🔍 Dry run - validating workflow...")
        valid, message = validate_workflow(yaml_file)
        if valid:
            print(f"✅ {message}")
            return True
        else:
            print(f"❌ Validation failed: {message}")
            return False
    
    print(f"🚀 Running workflow: {yaml_file}")
    
    try:
        results = execute_workflow(yaml_file, verbose=True)
        
        print("\n✅ Workflow completed successfully!")
        
        # Show what was created
        if yaml_file in EXAMPLES:
            creates = EXAMPLES[yaml_file].get('creates', [])
            if creates:
                print("\n📁 Files/directories created:")
                for item in creates:
                    status = "✅" if os.path.exists(item) else "❌"
                    print(f"   {status} {item}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Workflow failed: {str(e)}")
        return False


def cleanup_outputs():
    """Clean up output files and directories from examples."""
    cleanup_items = []
    for info in EXAMPLES.values():
        cleanup_items.extend(info.get('creates', []))
    
    # Add common output patterns
    cleanup_items.extend([
        "test_results.csv",
        "*.xlsx", 
        "*.json",
        "*_index/",
        "test_*"
    ])
    
    cleaned = []
    for pattern in cleanup_items:
        if '*' in pattern:
            # Glob pattern
            for item in glob.glob(pattern):
                if os.path.exists(item):
                    if os.path.isdir(item):
                        import shutil
                        shutil.rmtree(item)
                        cleaned.append(f"directory {item}")
                    else:
                        os.remove(item)
                        cleaned.append(f"file {item}")
        else:
            # Direct path
            if os.path.exists(pattern):
                if os.path.isdir(pattern):
                    import shutil
                    shutil.rmtree(pattern)
                    cleaned.append(f"directory {pattern}")
                else:
                    os.remove(pattern)
                    cleaned.append(f"file {pattern}")
    
    if cleaned:
        print(f"🧹 Cleaned up: {', '.join(cleaned)}")
    else:
        print("🧹 Nothing to clean up")


def main():
    parser = argparse.ArgumentParser(
        description="Workflow example runner and tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_examples.py --list
  python run_examples.py --run example_workflow.yaml
  python run_examples.py --validate all  
  python run_examples.py --dry-run advanced_workflow_example.yaml
  python run_examples.py --cleanup
        """
    )
    
    parser.add_argument('--list', action='store_true', help='List all available examples')
    parser.add_argument('--run', metavar='FILE', help='Run a specific workflow example')
    parser.add_argument('--validate', metavar='FILE', help='Validate workflow(s) without running ("all" for all)')
    parser.add_argument('--dry-run', metavar='FILE', help='Validate workflow without executing')
    parser.add_argument('--cleanup', action='store_true', help='Clean up output files from examples')
    
    args = parser.parse_args()
    
    if args.list:
        list_examples()
    
    elif args.run:
        yaml_file = args.run
        if not yaml_file.endswith('.yaml'):
            # Allow shorthand names
            matches = [f for f in EXAMPLES.keys() if yaml_file in f]
            if matches:
                yaml_file = matches[0]
                print(f"Using: {yaml_file}")
            else:
                print(f"❌ No workflow found matching '{args.run}'")
                return 1
        
        if not os.path.exists(yaml_file):
            print(f"❌ File not found: {yaml_file}")
            return 1
        
        success = run_workflow(yaml_file)
        return 0 if success else 1
    
    elif args.validate:
        if args.validate == 'all':
            files = list(EXAMPLES.keys())
        else:
            files = [args.validate]
        
        success_count = 0
        for yaml_file in files:
            if not os.path.exists(yaml_file):
                print(f"❌ File not found: {yaml_file}")
                continue
            
            valid, message = validate_workflow(yaml_file)
            status = "✅" if valid else "❌"
            print(f"{status} {yaml_file}: {message}")
            if valid:
                success_count += 1
        
        print(f"\n📊 Validation summary: {success_count}/{len(files)} workflows valid")
        return 0 if success_count == len(files) else 1
    
    elif args.dry_run:
        yaml_file = args.dry_run
        if not os.path.exists(yaml_file):
            print(f"❌ File not found: {yaml_file}")
            return 1
        
        success = run_workflow(yaml_file, dry_run=True)
        return 0 if success else 1
    
    elif args.cleanup:
        cleanup_outputs()
    
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())