#!/usr/bin/env python3
"""
Improved PDF slide extraction with title extraction and content analysis
- Extracts title from first page
- Skips title page in slides
- Detects and tags figures/equations
- Adds content metadata
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional

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
    text = text.strip()
    
    return text


def extract_title_from_first_page(text: str) -> str:
    """Extract lecture title from first page"""
    if not text:
        return "Unknown Lecture"
    
    # Clean the text
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Look for course number pattern (ELEC576, COMP576, etc.)
    title_candidates = []
    
    for line in lines[:10]:  # Check first 10 lines
        # Skip common header patterns
        if any(skip in line.lower() for skip in ['baylor', 'college', 'medicine', 'rice university', 'ece dept', 'cs dept']):
            continue
        
        # Look for course title patterns
        if '576' in line or 'deep' in line.lower() or 'machine learning' in line.lower():
            # Clean up the line
            cleaned = re.sub(r'ELEC/COMP\s*\d+:\s*', '', line, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
            if cleaned and len(cleaned) > 5:
                title_candidates.append(cleaned)
    
    # Return the first good candidate, or use first substantial line
    if title_candidates:
        return title_candidates[0]
    
    # Fallback: use first substantial line
    for line in lines:
        if len(line) > 10 and not any(skip in line.lower() for skip in ['ankit', 'patel', 'baylor', 'rice']):
            return line[:100]  # Limit length
    
    return "Unknown Lecture"


def analyze_slide_content(text: str) -> Dict:
    """Analyze slide content to detect figures, equations, and topics"""
    analysis = {
        'has_equations': False,
        'has_figures': False,
        'content_type': 'text',  # 'text', 'figure', 'equation', 'mixed'
        'topics': [],
        'is_title_slide': False
    }
    
    if not text or len(text) < 20:
        analysis['content_type'] = 'minimal'
        return analysis
    
    text_lower = text.lower()
    
    # Detect equations (common patterns)
    equation_patterns = [
        r'[=<>≤≥≠]',  # Math operators
        r'[∑∏∫∂∇]',  # Math symbols
        r'\d+\s*[+\-*/]\s*\d+',  # Simple math
        r'[a-z]\s*=\s*[a-z]',  # Variable assignments
        r'\([^)]*\)\s*[=<>]',  # Parentheses with operators
    ]
    
    equation_count = sum(len(re.findall(pattern, text)) for pattern in equation_patterns)
    if equation_count > 3:
        analysis['has_equations'] = True
    
    # Detect figure references
    figure_patterns = [
        r'figure\s+\d+',
        r'fig\.\s*\d+',
        r'image',
        r'diagram',
        r'plot',
        r'graph',
        r'chart'
    ]
    
    figure_count = sum(len(re.findall(pattern, text_lower)) for pattern in figure_patterns)
    if figure_count > 0:
        analysis['has_figures'] = True
    
    # Detect title slide
    if any(word in text_lower for word in ['introduction', 'overview', 'agenda', 'outline']) and len(text.split()) < 30:
        analysis['is_title_slide'] = True
    
    # Determine content type
    if analysis['has_equations'] and analysis['has_figures']:
        analysis['content_type'] = 'mixed'
    elif analysis['has_equations']:
        analysis['content_type'] = 'equation'
    elif analysis['has_figures']:
        analysis['content_type'] = 'figure'
    
    # Extract topics (simple keyword detection)
    topic_keywords = {
        'neural networks': ['neural network', 'nn', 'perceptron', 'activation'],
        'backpropagation': ['backpropagation', 'backprop', 'gradient descent'],
        'cnn': ['convolutional', 'cnn', 'conv', 'pooling'],
        'rnn': ['recurrent', 'rnn', 'lstm', 'gru'],
        'optimization': ['optimization', 'adam', 'sgd', 'momentum', 'learning rate'],
        'regularization': ['regularization', 'dropout', 'batch norm', 'l1', 'l2'],
        'loss functions': ['loss', 'cross entropy', 'mse', 'mae'],
        'attention': ['attention', 'transformer', 'self-attention'],
        'generative models': ['gan', 'vae', 'generative', 'autoencoder'],
        'transfer learning': ['transfer learning', 'fine-tuning', 'pretrained']
    }
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            analysis['topics'].append(topic)
    
    return analysis


def extract_text_from_pdf(pdf_path: Path) -> Optional[Dict]:
    """Extract text from a single PDF file with improved processing"""
    print(f"  Processing {pdf_path.name}...", end=' ')
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            # Extract title from first page
            first_page = pdf.pages[0]
            first_page_text = first_page.extract_text() or ''
            title = extract_title_from_first_page(first_page_text)
            
            slides_data = {
                'filename': pdf_path.name,
                'lecture_number': extract_lecture_number(pdf_path.name),
                'title': title,
                'total_pages': total_pages,
                'slides': []
            }
            
            # Process slides (skip first page)
            for page_num, page in enumerate(pdf.pages[1:], start=2):  # Start from page 2
                text = page.extract_text()
                
                if text:
                    cleaned_text = clean_text(text)
                    
                    if cleaned_text and len(cleaned_text) > 10:
                        # Analyze content
                        analysis = analyze_slide_content(cleaned_text)
                        
                        slide_entry = {
                            'page': page_num,
                            'text': cleaned_text,
                            'content_type': analysis['content_type'],
                            'has_equations': analysis['has_equations'],
                            'has_figures': analysis['has_figures'],
                            'topics': analysis['topics']
                        }
                        
                        # Add summary for figure/equation slides
                        if analysis['content_type'] in ['figure', 'equation', 'mixed']:
                            # Extract key phrases from surrounding text
                            sentences = re.split(r'[.!?]\s+', cleaned_text)
                            key_sentences = [s.strip() for s in sentences if len(s.strip()) > 20 and len(s.strip()) < 200]
                            if key_sentences:
                                slide_entry['summary'] = key_sentences[0] if key_sentences else None
                        
                        slides_data['slides'].append(slide_entry)
            
            print(f"✓ {len(slides_data['slides'])} slides (title: {title[:50]}...)")
            return slides_data
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_lecture_number(filename: str) -> str:
    """Extract lecture number from filename"""
    # Pattern: ELEC576-Lec01.pdf -> "01"
    match = re.search(r'Lec(\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Special case: ELEC576_NNasSpine.pdf
    if 'NNasSpine' in filename:
        return 'NNasSpine'
    
    return 'unknown'


def main():
    print("🚀 Improved PDF slide extraction...\n")
    
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
    content_type_counts = {'text': 0, 'figure': 0, 'equation': 0, 'mixed': 0, 'minimal': 0}
    
    # Process each PDF
    for pdf_file in pdf_files:
        slides_data = extract_text_from_pdf(pdf_file)
        
        if slides_data:
            all_slides.append(slides_data)
            total_slides += len(slides_data['slides'])
            total_chars += sum(len(s['text']) for s in slides_data['slides'])
            
            # Count content types
            for slide in slides_data['slides']:
                content_type = slide.get('content_type', 'text')
                content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
            
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
            'content_type_distribution': content_type_counts,
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
    
    print(f"\n📈 Content Type Distribution:")
    for content_type, count in content_type_counts.items():
        percentage = (count / total_slides * 100) if total_slides > 0 else 0
        print(f"   {content_type}: {count} ({percentage:.1f}%)")
    
    print(f"\n💾 Output files:")
    print(f"   Individual files: {output_dir}/*.json")
    print(f"   Combined file: {combined_file}")
    
    # Show sample
    if all_slides:
        print(f"\n📝 Sample lecture: {all_slides[0]['title']}")
        if all_slides[0]['slides']:
            sample = all_slides[0]['slides'][0]
            print(f"   Page {sample['page']}: {sample['text'][:100]}...")
            print(f"   Content type: {sample['content_type']}, Topics: {sample.get('topics', [])}")


if __name__ == '__main__':
    main()

