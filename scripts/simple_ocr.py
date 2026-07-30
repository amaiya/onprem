#!/usr/bin/env python3
"""
Standalone OCR Script using unstructured-io library
"""

import sys
import os
import argparse
from pathlib import Path


def ocr_document(input_path, output_path=None, mode="elements", strategy="hi_res", 
                 model_name=None, infer_tables=False, include_page_breaks=True):
    """
    Perform OCR on a document using unstructured library.
    
    Args:
        input_path: Path to input file (PDF or image)
        output_path: Optional path to output text file
        mode: Processing mode ("single", "elements", or "paged")
        strategy: Processing strategy ("fast", "hi_res", "ocr_only")
        model_name: Custom model name or path for layout detection
        infer_tables: Whether to infer table structure
        include_page_breaks: Whether to include page break markers
    
    Returns:
        Extracted text as string
    """
    try:
        from unstructured.partition.auto import partition
    except ImportError:
        print("Error: unstructured library not found.")
        print("Install with: pip install unstructured[all-docs]")
        sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    print(f"Processing: {input_path}")
    print(f"Mode: {mode}, Strategy: {strategy}")
    if model_name:
        print(f"Model: {model_name}")
    
    try:
        # Partition the document
        partition_kwargs = {
            'filename': input_path,
            'strategy': strategy,
            'infer_table_structure': infer_tables,
            'include_page_breaks': include_page_breaks,
        }
        
        # Add model name if specified
        if model_name:
            partition_kwargs['hi_res_model_name'] = model_name
        
        elements = partition(**partition_kwargs)
        
        # Extract text from elements
        text_parts = []
        for element in elements:
            text_parts.append(str(element))
        
        full_text = "\n\n".join(text_parts)
        
        print(f"Extracted {len(text_parts)} elements")
        print(f"Total characters: {len(full_text)}")
        
        # Save to file if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"Output saved to: {output_path}")
        
        return full_text
        
    except Exception as e:
        print(f"Error processing document: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for command line usage"""
    
    parser = argparse.ArgumentParser(
        description='Standalone OCR Script using unstructured-io library',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf
  %(prog)s document.pdf output.txt
  %(prog)s scanned_page.jpg --strategy ocr_only
  %(prog)s input.pdf -o output.txt --tables
  %(prog)s input.pdf --model yolox_tiny  # Use smaller/faster model

Dependencies:
  pip install unstructured[all-docs]
  
  Or minimal install:
  pip install unstructured "unstructured[pdf]" pillow pytesseract
  
  System dependencies:
  - tesseract-ocr (for OCR)
  - poppler-utils (for PDF processing)


Offline Usage:

  Method 1 - Copy from internet machine (Recommended):
    1. On internet machine, run the pipeline to populate the cache:
       python simple_ocr.py sample.pdf
    
    2. Tar the folder while PRESERVING symlinks (Crucial: use the -h flag to follow/archive target data, or omit it but ensure your tar flags preserve links):
       tar -chzf models.tar.gz -C ~/.cache/huggingface/hub/ models--unstructuredio--yolo_x_layout
    
    3. Transfer models.tar.gz to the offline machine.
    
    4. On the offline machine, extract it:
       mkdir -p ~/.cache/huggingface/hub
       tar -xzf models.tar.gz -C ~/.cache/huggingface/hub/

  Method 2 - Manual download (Simple "No-Symlink" Hack):
    Instead of recreating the highly temperamental internal blobs layout, you can trick Hugging Face by placing the raw downloads directly into a dummy snapshot folder.

    1. Create the explicit main snapshot directory:
       mkdir -p ~/.cache/huggingface/hub/models--unstructuredio--yolo_x_layout/snapshots/main
       mkdir -p ~/.cache/huggingface/hub/models--unstructuredio--yolo_x_layout/refs

    2. Download the model file AND its repo configurations into that exact snapshot folder:
       cd ~/.cache/huggingface/hub/models--unstructuredio--yolo_x_layout/snapshots/main
       wget https://huggingface.co/unstructuredio/yolo_x_layout/resolve/main/yolox_l0.05.onnx
       wget https://huggingface.co

    3. Create the pointer reference file:
       echo "main" > ../../refs/main

    4. Set Python environment variable before running your offline script:
       export HF_HUB_OFFLINE=1

        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to input file (PDF or image)'
    )
    
    parser.add_argument(
        'output_file',
        nargs='?',
        help='Path to output text file (default: <input>_ocr.txt)'
    )
    
    parser.add_argument(
        '-o', '--output',
        dest='output_file_alt',
        help='Alternative way to specify output file'
    )
    
    parser.add_argument(
        '--strategy',
        choices=['fast', 'hi_res', 'ocr_only'],
        default='hi_res',
        help='Processing strategy: fast=text extraction only (no models), '
             'hi_res=layout detection + auto OCR (~207MB yolox_layout, default), '
             'ocr_only=force OCR + layout detection'
    )
    
    parser.add_argument(
        '--model',
        dest='model_name',
        help='Model name for layout detection. '
             'Options: yolox (default, ~207MB), yolox_tiny (~100MB), yolox_quantized. '
             'Note: Only predefined model names are supported, not custom file paths.'
    )
    
    parser.add_argument(
        '--tables',
        action='store_true',
        help='Enable table structure inference as HTML '
             '(downloads ~47MB resnet18 + ~111MB table-transformer models)'
    )
    
    parser.add_argument(
        '--no-page-breaks',
        action='store_true',
        help='Disable page break markers'
    )
    
    args = parser.parse_args()
    
    input_path = args.input_file
    
    # Determine output path
    output_path = args.output_file or args.output_file_alt
    if not output_path:
        # Default output: same name as input with .txt extension
        input_stem = Path(input_path).stem
        output_path = f"{input_stem}_ocr.txt"
    
    # Process document
    text = ocr_document(
        input_path,
        output_path,
        strategy=args.strategy,
        model_name=args.model_name,
        infer_tables=args.tables,
        include_page_breaks=not args.no_page_breaks
    )
    
    # Print preview
    print("\n" + "="*60)
    print("TEXT PREVIEW (first 500 characters):")
    print("="*60)
    print(text[:500])
    if len(text) > 500:
        print("\n[... truncated ...]")


if __name__ == "__main__":
    main()
