#!/usr/bin/env python3
"""
Extract text from PDF lecture slides
Processes all PDFs in data/raw/slides/ and saves to data/processed/slides/
"""

import json
import re
from pathlib import Path
from typing import List, Dict

try:
    import pdfplumber
except ImportError:
    print("❌ Missing pdfplumber. Install: pip install pdfplumber")
    exit(1)


def clean_text(text: str) -> str:
    """Clean extracted text"""
    if not text:
        return ''
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and headers/footers (common patterns)
    text = re.sub(r'Page \d+', '', text)
    text = re.sub(r'\d+\s*/\s*\d+', '', text)  # "1 / 10" style page numbers
    
    # Clean up
    text = text.strip()
    
    return text


def extract_text_from_pdf(pdf_path: Path) -> Dict:
    """Extract text from a single PDF file"""
    print(f"  Processing {pdf_path.name}...", end=' ')
    
    slides_data = {
        'filename': pdf_path.name,
        'lecture_number': extract_lecture_number(pdf_path.name),
        'total_pages': 0,
        'slides': []
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            slides_data['total_pages'] = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                text = page.extract_text()
                
                if text:
                    cleaned_text = clean_text(text)
                    
                    if cleaned_text and len(cleaned_text) > 10:  # Skip empty/minimal pages
                        slide_entry = {
                            'page': page_num,
                            'text': cleaned_text,
                            'char_count': len(cleaned_text),
                            'word_count': len(cleaned_text.split())
                        }
                        slides_data['slides'].append(slide_entry)
            
            print(f"✓ {len(slides_data['slides'])} slides extracted")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    return slides_data


def extract_lecture_number(filename: str) -> str:
    """Extract lecture number from filename"""
    # Pattern: ELEC576-Lec01.pdf -> "01"
    match = re.search(r'Lec(\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Pattern: ELEC576-Lec11.pdf -> "11"
    match = re.search(r'lec(\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Special case: ELEC576_NNasSpine.pdf
    if 'NNasSpine' in filename:
        return 'NNasSpine'
    
    return 'unknown'


def main():
    print("🚀 Extracting text from PDF slides...\n")
    
    # Paths
    input_dir = Path('data/raw/slides')
    output_dir = Path('data/processed/slides')
    
    if not input_dir.exists():
        print(f"❌ Slides directory not found: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = sorted(input_dir.glob('*.pdf'))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {input_dir}")
        return
    
    print(f"📚 Found {len(pdf_files)} PDF files\n")
    
    all_slides = []
    total_slides = 0
    total_chars = 0
    
    # Process each PDF
    for pdf_file in pdf_files:
        slides_data = extract_text_from_pdf(pdf_file)
        
        if slides_data:
            all_slides.append(slides_data)
            total_slides += len(slides_data['slides'])
            total_chars += sum(s['char_count'] for s in slides_data['slides'])
            
            # Save individual file
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(slides_data, f, indent=2, ensure_ascii=False)
    
    # Save combined file
    combined_file = output_dir / 'all_slides.json'
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_lectures': len(all_slides),
            'total_slides': total_slides,
            'total_characters': total_chars,
            'lectures': all_slides
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Done!")
    print(f"{'='*60}")
    print(f"\n📊 Statistics:")
    print(f"   Lectures processed: {len(all_slides)}")
    print(f"   Total slides: {total_slides}")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Avg slides per lecture: {total_slides / len(all_slides):.1f}")
    print(f"   Avg chars per slide: {total_chars / total_slides:.0f}")
    
    print(f"\n💾 Output files:")
    print(f"   Individual files: {output_dir}/*.json")
    print(f"   Combined file: {combined_file}")
    
    # Show sample
    if all_slides:
        print(f"\n📝 Sample (first slide from {all_slides[0]['filename']}):")
        if all_slides[0]['slides']:
            sample = all_slides[0]['slides'][0]['text']
            print(f"   {sample[:150]}...")


if __name__ == '__main__':
    main()
