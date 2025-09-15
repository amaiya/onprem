#!/usr/bin/env python3
"""
Complete FAR Legal Analysis Pipeline
Based on 99j_examples_legal_analysis.ipynb

This script:
1. Downloads FAR HTML files
2. Extracts Part 9 sections
3. Runs workflow to extract statutory citations
4. Outputs results to Excel spreadsheet
"""

import tempfile
import zipfile
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from onprem import utils as U
from onprem.ingest import extract_files
from onprem.pipelines.workflow import execute_workflow

def prepare_far_data():
    """Download and extract FAR data."""
    print("📥 Downloading FAR data...")
    
    # Download URL
    url = "https://www.acquisition.gov/sites/default/files/current/far/zip/html/FARHTML.zip"
    
    # Create data directory
    data_dir = "/tmp/far_data"
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "FARHTML.zip")
    
    # Download the ZIP file
    U.download(url, zip_path, verify=True)
    print(f"✅ Downloaded to {zip_path}")
    
    # Extract the ZIP file
    print("📂 Extracting ZIP file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Find Part 9 files
    all_files = list(extract_files(data_dir))
    part9_files = [
        fname for fname in all_files 
        if fname.lower().endswith('.html') and 
           os.path.basename(fname).startswith('9.')
    ]
    
    print(f"📊 Total FAR files: {len(all_files)}")
    print(f"📊 Part 9 sections: {len(part9_files)}")
    print("📁 Sample Part 9 files:")
    for fname in part9_files[:5]:
        print(f"   {os.path.basename(fname)}")
    
    return data_dir, len(part9_files)

def run_analysis():
    """Run the FAR legal analysis workflow."""
    print("\n🔍 Running statute extraction analysis...")
    
    # Change to workflow directory
    workflow_dir = Path(__file__).parent / "workflows"
    os.chdir(workflow_dir)
    
    try:
        # Execute the workflow
        results = execute_workflow("yaml_examples/far_legal_analysis_simple.yaml", verbose=True)
        print("✅ Analysis completed successfully!")
        
        # Check if output file was created
        output_file = "far_statutory_citations.xlsx"
        if os.path.exists(output_file):
            print(f"📊 Results saved to: {os.path.abspath(output_file)}")
        else:
            print("⚠️  Output file not found")
            
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def main():
    """Main execution function."""
    print("🏛️  FAR Legal Analysis Pipeline")
    print("=" * 50)
    
    try:
        # Step 1: Prepare data
        data_dir, num_files = prepare_far_data()
        
        if num_files == 0:
            print("❌ No Part 9 files found!")
            return 1
        
        # Step 2: Run analysis
        success = run_analysis()
        
        if success:
            print("\n🎉 Analysis pipeline completed successfully!")
            print("\nResults summary:")
            print("- FAR Part 9 sections analyzed for statutory citations")
            print("- Results exported to Excel spreadsheet")
            print("- Use spreadsheet to identify regulation-heavy vs statute-driven sections")
            return 0
        else:
            print("\n❌ Analysis pipeline failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())