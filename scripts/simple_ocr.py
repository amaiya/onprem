#!/usr/bin/env python3
"""
Standalone OCR Script using Tesseract only
"""

import sys
import os
import argparse
from pathlib import Path


def ocr_document(input_path, output_path=None, lang='eng', psm=3, dpi=300):
    """
    Perform OCR on a document using Tesseract.
    
    Args:
        input_path: Path to input file (PDF or image)
        output_path: Optional path to output text file
        lang: Tesseract language code (default: 'eng')
        psm: Page segmentation mode (default: 3 - fully automatic)
        dpi: DPI for PDF rendering (default: 300)
    
    Returns:
        Extracted text as string
    """
    try:
        import pytesseract
        from PIL import Image
        from pdf2image import convert_from_path
    except ImportError as e:
        print(f"Error: Required Python library not found: {e}")
        print("Install with: pip install pytesseract pillow pdf2image")
        sys.exit(1)
    
    # Check if tesseract is installed
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        print("Error: Tesseract OCR not found in system PATH")
        print("Install tesseract:")
        print("  Ubuntu/Debian: apt-get install tesseract-ocr")
        print("  macOS: brew install tesseract")
        sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    print(f"Processing: {input_path}")
    print(f"Language: {lang}, PSM: {psm}, DPI: {dpi}")
    
    try:
        # Determine file type
        file_ext = Path(input_path).suffix.lower()
        
        if file_ext == '.pdf':
            # Handle PDF files
            print("Converting PDF to images...")
            images = convert_from_path(input_path, dpi=dpi)
            print(f"Converted {len(images)} page(s)")
            
            # OCR each page
            text_parts = []
            for i, image in enumerate(images, 1):
                print(f"Processing page {i}/{len(images)}...")
                
                # Configure Tesseract
                custom_config = f'--psm {psm}'
                
                # Perform OCR
                page_text = pytesseract.image_to_string(
                    image,
                    lang=lang,
                    config=custom_config
                )
                
                text_parts.append(f"--- Page {i} ---\n{page_text}")
            
            full_text = "\n\n".join(text_parts)
            
        else:
            # Handle image files directly
            print("Processing image...")
            image = Image.open(input_path)
            
            # Configure Tesseract
            custom_config = f'--psm {psm}'
            
            # Perform OCR
            full_text = pytesseract.image_to_string(
                image,
                lang=lang,
                config=custom_config
            )
        
        print(f"Total characters extracted: {len(full_text)}")
        
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
        description='Standalone OCR Script using Tesseract only',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf
  %(prog)s document.pdf output.txt
  %(prog)s scanned_page.jpg
  %(prog)s input.pdf -o output.txt --lang eng
  %(prog)s input.pdf --dpi 600 --psm 6

Page Segmentation Modes (PSM):
  0  = Orientation and script detection (OSD) only
  1  = Automatic page segmentation with OSD
  2  = Automatic page segmentation, but no OSD, or OCR
  3  = Fully automatic page segmentation, but no OSD (default)
  4  = Assume a single column of text of variable sizes
  5  = Assume a single uniform block of vertically aligned text
  6  = Assume a single uniform block of text
  7  = Treat the image as a single text line
  8  = Treat the image as a single word
  9  = Treat the image as a single word in a circle
  10 = Treat the image as a single character
  11 = Sparse text. Find as much text as possible in no particular order
  12 = Sparse text with OSD
  13 = Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific

Dependencies:
  pip install pytesseract pillow pdf2image
  
System dependencies:
  - tesseract-ocr (for OCR)
    Ubuntu/Debian: apt-get install tesseract-ocr
    macOS: brew install tesseract
    
  - poppler-utils (for PDF processing)
    Ubuntu/Debian: apt-get install poppler-utils
    macOS: brew install poppler

Language Support:
  To use languages other than English, install language data:
    Ubuntu/Debian: apt-get install tesseract-ocr-[lang]
    Example: apt-get install tesseract-ocr-spa (Spanish)
    
  Common language codes: eng, spa, fra, deu, ita, por, rus, chi_sim, chi_tra, jpn, ara
  List installed languages: tesseract --list-langs
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
        '--lang',
        default='eng',
        help='Tesseract language code (default: eng). Use + for multiple languages, e.g., eng+fra'
    )
    
    parser.add_argument(
        '--psm',
        type=int,
        default=3,
        choices=range(0, 14),
        help='Page segmentation mode (default: 3). See PSM list above for details.'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for PDF rendering (default: 300). Higher values improve quality but increase processing time.'
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
        lang=args.lang,
        psm=args.psm,
        dpi=args.dpi
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
