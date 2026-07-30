#!/usr/bin/env python3
"""
Standalone OCR Script using unstructured-io library
"""

import sys
import os
import argparse
from pathlib import Path


def ocr_document(input_path, output_path=None, mode="elements", strategy="hi_res", 
                 infer_tables=False, include_page_breaks=True):
    """
    Perform OCR on a document using unstructured library.
    
    Args:
        input_path: Path to input file (PDF or image)
        output_path: Optional path to output text file
        mode: Processing mode ("single", "elements", or "paged")
        strategy: Processing strategy ("fast", "hi_res", "ocr_only")
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
    
    try:
        # Partition the document
        elements = partition(
            filename=input_path,
            strategy=strategy,
            infer_table_structure=infer_tables,
            include_page_breaks=include_page_breaks,
        )
        
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

Dependencies:
  pip install unstructured[all-docs]
  
  Or minimal install:
  pip install unstructured "unstructured[pdf]" pillow pytesseract
  
  System dependencies:
  - tesseract-ocr (for OCR)
  - poppler-utils (for PDF processing)
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
