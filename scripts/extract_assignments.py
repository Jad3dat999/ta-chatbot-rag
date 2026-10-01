#!/usr/bin/env python3
"""
Extract text from assignment PDFs
Processes all PDFs in data/raw/assignment/ and saves to data/processed/assignments/
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


def extract_assignment_number(filename: str) -> str:
    """Extract assignment number from filename"""
    # Pattern: ELEC576_Assignment_0.pdf -> "0"
    match = re.search(r'Assignment[_\s]*(\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return 'unknown'


def detect_sections(text: str) -> List[Dict]:
    """Detect sections in assignment text based on common patterns"""
    sections = []
    
    # Find all section boundaries using regex
    # Pattern: Numbered sections like "1 Python Machine Learning Stack" or "2 Interactive Terminal"
    # Match: space + number (1-20) + space + capital letter + words (up to next number or end)
    # Exclude common header words like "Assignment", "Due", "Fall", etc.
    excluded_words = ['Assignment', 'Due', 'Fall', 'Spring', 'Winter', 'Summer', 'ELEC', 'COMP']
    # Match section numbers - we'll extract titles separately
    # Pattern: space + number (1-20) + space + capital letter
    section_pattern = r'\s([1-9]|1[0-9]|20)\s+([A-Z])'
    
    matches = []
    for match in re.finditer(section_pattern, text):
        section_num = int(match.group(1))
        
        # Skip if this is a page number
        context_before = text[max(0, match.start()-50):match.start()]
        context_after = text[match.end():match.end()+30]
        
        # Pattern 1: "Assignment 0 5" (page number)
        if re.search(r'Assignment\s+\d+\s+\d+\s*$', context_before):
            continue
        
        # Pattern 2: "Assignment 0" followed by newline/page break, then number + "Submission" (page number)
        if 'Assignment' in context_before and re.search(r'^\s+(Submission|Collaboration|Plagiarism)', context_after, re.IGNORECASE):
            continue
        
        # Pattern 3: Page footer pattern "– Assignment 0 5"
        if re.search(r'–\s*Assignment\s+\d+\s+\d+\s*$', context_before):
            continue
        
        # Extract full title by looking ahead
        # Title usually ends when content starts (sentence beginning)
        # Titles typically end with: closing paren, or before a sentence that starts content
        title_region = text[match.start():match.start() + 150]  # Look ahead 150 chars
        
        # Find where title ends - titles usually end with:
        # 1. Closing parenthesis: "Stack (Anaconda)"
        # 2. Before a sentence that starts content: "You will", "The assignment", "In order"
        # Look for sentence patterns that indicate content start
        content_patterns = [
            r'([A-Z][a-z]+\s+(?:will|can|should|must|is|are|has|have)\s+)',  # "You will", "It can"
            r'([A-Z][a-z]+\s+(?:the|a|an|this|that)\s+[a-z]+)',  # "The assignment", "This course"
            r'([A-Z][a-z]+\s+[a-z]+\s+(?:in|to|for|with|from))',  # "In order", "To prepare"
        ]
        
        title_end_pos = None
        for pattern in content_patterns:
            content_match = re.search(pattern, title_region)
            if content_match:
                title_end_pos = content_match.start()
                break
        
        if title_end_pos:
            # Extract title from number to content start
            title_text = text[match.start():match.start() + title_end_pos]
            # Extract just the title part (after number)
            title_match = re.search(r'\d+\s+(.+)', title_text)
            if title_match:
                section_title = title_match.group(1).strip()
                # Clean up: remove trailing spaces, keep parentheses
                section_title = re.sub(r'\s+', ' ', section_title).strip()
            else:
                section_title = match.group(2)
        else:
            # Fallback: try to find title ending at closing paren or reasonable length
            # Look for pattern: number + title (up to closing paren or ~80 chars)
            title_match = re.search(r'\d+\s+([A-Z][A-Za-z\s()]{5,80}?)(?:\)|$)', title_region)
            if title_match:
                section_title = title_match.group(1).strip()
            else:
                # Last resort: use first 60 chars after number
                title_match = re.search(r'\d+\s+([A-Z][A-Za-z\s()]{5,60})', title_region)
                section_title = title_match.group(1).strip() if title_match else match.group(2)
        
        # Skip if title starts with excluded words
        if any(section_title.startswith(word) for word in excluded_words):
            continue
        
        # Skip if title starts with named section keywords (will be handled separately)
        if any(section_title.startswith(word) for word in ['Submission', 'Collaboration', 'Plagiarism']):
            continue
        
        # Store match with title
        matches.append((match, section_title))
    
    if matches:
        # First section: everything before first numbered section
        first_match, _ = matches[0]
        intro_text = text[:first_match.start()].strip()
        if intro_text and len(intro_text) > 50:
            sections.append({
                'title': 'Introduction',
                'content': clean_text(intro_text)
            })
        
        # Process numbered sections
        for i, (match, improved_title) in enumerate(matches):
            section_num = match.group(1)
            section_title = improved_title if improved_title else match.group(2).strip()
            full_title = f"{section_num} {section_title}"
            
            # Find content: from end of this match to start of next match
            start_pos = match.end()
            end_pos = matches[i + 1][0].start() if i + 1 < len(matches) else len(text)
            
            content = text[start_pos:end_pos].strip()
            content = clean_text(content)
            
            if content and len(content) > 20:
                sections.append({
                    'title': full_title,
                    'content': content
                })
    
    # Find named sections at the end (Submission Instructions, Collaboration Policy, Plagiarism)
    named_sections = [
        (r'(Submission\s+Instructions?)', 'Submission Instructions'),
        (r'(Collaboration\s+Policy)', 'Collaboration Policy'),
        (r'(Plagiarism)', 'Plagiarism'),
    ]
    
    for pattern, default_title in named_sections:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Find where this section starts
            start_pos = match.start()
            
            # Find where next section starts or end of text
            end_pos = len(text)
            for p, _ in named_sections:
                next_match = re.search(p, text[start_pos + 50:], re.IGNORECASE)
                if next_match:
                    potential_end = start_pos + 50 + next_match.start()
                    if potential_end < end_pos:
                        end_pos = potential_end
            
            # Extract section title and content
            section_text = text[start_pos:end_pos]
            # Title is usually the first line or first 100 chars
            lines = section_text.split('\n')
            title_line = lines[0].strip() if lines else section_text[:100].strip()
            
            # Content starts after title
            content_start = len(title_line) if title_line in section_text[:200] else 0
            content = section_text[content_start:].strip()
            content = clean_text(content)
            
            if content and len(content) > 20:
                # Check if this section already exists (might overlap with numbered sections)
                title_to_use = title_line if len(title_line) < 100 else default_title
                if not any(s['title'].lower() == title_to_use.lower() for s in sections):
                    sections.append({
                        'title': title_to_use,
                        'content': content
                    })
    
    # If no sections found, create one default section
    if not sections:
        sections.append({
            'title': 'Full Assignment',
            'content': clean_text(text)
        })
    
    return sections


def extract_text_from_pdf(pdf_path: Path) -> Optional[Dict]:
    """Extract text from a single assignment PDF"""
    print(f"  Processing {pdf_path.name}...", end=' ')
    
    assignment_data = {
        'filename': pdf_path.name,
        'assignment_number': extract_assignment_number(pdf_path.name),
        'total_pages': 0,
        'sections': [],
        'full_text': ''
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            assignment_data['total_pages'] = len(pdf.pages)
            
            # Extract text from all pages
            all_text_parts = []
            
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    cleaned_text = clean_text(text)
                    if cleaned_text:
                        all_text_parts.append(cleaned_text)
            
            # Combine all text
            full_text = '\n'.join(all_text_parts)
            assignment_data['full_text'] = full_text
            
            # Detect sections
            sections = detect_sections(full_text)
            assignment_data['sections'] = sections
            
            print(f"✓ {len(pdf.pages)} pages, {len(sections)} sections")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return assignment_data


def main():
    print("🚀 Extracting text from assignment PDFs...\n")
    
    # Paths
    input_dir = Path('data/raw/assignment')
    output_dir = Path('data/processed/assignments')
    
    if not input_dir.exists():
        print(f"❌ Assignment directory not found: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = sorted(input_dir.glob('*.pdf'))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {input_dir}")
        return
    
    print(f"📚 Found {len(pdf_files)} PDF files\n")
    
    all_assignments = []
    total_pages = 0
    total_sections = 0
    
    # Process each PDF
    for pdf_file in pdf_files:
        assignment_data = extract_text_from_pdf(pdf_file)
        
        if assignment_data:
            all_assignments.append(assignment_data)
            total_pages += assignment_data['total_pages']
            total_sections += len(assignment_data['sections'])
            
            # Save individual file
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(assignment_data, f, indent=2, ensure_ascii=False)
    
    # Save combined file
    combined_file = output_dir / 'all_assignments.json'
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_assignments': len(all_assignments),
            'total_pages': total_pages,
            'total_sections': total_sections,
            'assignments': all_assignments
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Done!")
    print(f"{'='*60}")
    print(f"\n📊 Statistics:")
    print(f"   Assignments processed: {len(all_assignments)}")
    print(f"   Total pages: {total_pages}")
    print(f"   Total sections: {total_sections}")
    print(f"   Avg pages per assignment: {total_pages / len(all_assignments):.1f}")
    print(f"   Avg sections per assignment: {total_sections / len(all_assignments):.1f}")
    
    print(f"\n💾 Output files:")
    print(f"   Individual files: {output_dir}/*.json")
    print(f"   Combined file: {combined_file}")
    
    # Show sample
    if all_assignments:
        print(f"\n📝 Sample assignment: {all_assignments[0]['filename']}")
        print(f"   Assignment #{all_assignments[0]['assignment_number']}")
        print(f"   Sections: {len(all_assignments[0]['sections'])}")
        if all_assignments[0]['sections']:
            print(f"   First section: {all_assignments[0]['sections'][0]['title']}")
            print(f"   Content preview: {all_assignments[0]['sections'][0]['content'][:100]}...")


if __name__ == '__main__':
    main()

