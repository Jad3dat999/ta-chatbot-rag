#!/usr/bin/env python3
"""
Extract sections from assignment markdown files
Processes all .md files in data/raw/assignment/ and saves to data/processed/assignments/
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional


def clean_text(text: str) -> str:
    """Clean extracted text while preserving markdown formatting"""
    if not text:
        return ''
    
    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove horizontal rules (---) as they're just separators
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
    
    # Clean up excessive whitespace at start/end
    text = text.strip()
    
    return text


def extract_assignment_number(filename: str) -> str:
    """Extract assignment number from filename"""
    # Pattern: assignment0.md -> "0"
    match = re.search(r'assignment(\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return 'unknown'


def parse_markdown_sections(md_content: str) -> List[Dict]:
    """Parse markdown content into sections (tasks included in parent sections)"""
    sections = []
    lines = md_content.split('\n')
    
    current_section = None
    current_content = []
    intro_content = []
    found_first_section = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for main section headers (##) - NOT ### (those are tasks/subsections)
        if line.startswith('##') and not line.startswith('###'):
            # Skip the assignment title header (## Assignment 0)
            if re.match(r'^##\s+Assignment\s+\d+', line, re.IGNORECASE):
                i += 1
                continue
            
            # Save previous section (which should include any tasks)
            if current_section and current_content:
                content = '\n'.join(current_content).strip()
                if content and len(content) > 20:
                    sections.append({
                        'title': current_section,
                        'content': clean_text(content)
                    })
            
            # Start new section
            section_title = line.lstrip('#').strip()
            # Keep numbering in title (e.g., "1. Python Machine Learning Stack")
            current_section = section_title
            current_content = []
            found_first_section = True
        elif line.startswith('###'):
            # Task/subsection - ALWAYS include in current section content (never create new section)
            if current_section:
                current_content.append(line)  # Include the task header
            # If no current section, skip (shouldn't happen, but handle gracefully)
        elif not found_first_section and line.strip() and not line.startswith('#'):
            # Content before first numbered section (introduction)
            # Skip horizontal rules and empty lines
            if not line.strip() == '---':
                intro_content.append(line)
        else:
            # Add to current section content (including tasks)
            if current_section:
                current_content.append(line)
        
        i += 1
    
    # Save introduction if exists
    if intro_content:
        intro_text = '\n'.join(intro_content).strip()
        if intro_text and len(intro_text) > 20:
            sections.insert(0, {
                'title': 'Introduction',
                'content': clean_text(intro_text)
            })
    
    # Save last section
    if current_section and current_content:
        content = '\n'.join(current_content).strip()
        if content and len(content) > 20:
            sections.append({
                'title': current_section,
                'content': clean_text(content)
            })
    
    return sections


def extract_text_from_markdown(md_path: Path) -> Optional[Dict]:
    """Extract sections from a markdown file"""
    print(f"  Processing {md_path.name}...", end=' ')
    
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Extract assignment title from "## Assignment X" header
        title_match = re.search(r'^##\s+Assignment\s+(\d+)', md_content, re.MULTILINE | re.IGNORECASE)
        if title_match:
            assignment_title = f"Assignment {title_match.group(1)}"
        else:
            # Fallback: use first ## header
            title_match = re.search(r'^##\s+(.+)$', md_content, re.MULTILINE)
            assignment_title = title_match.group(1).strip() if title_match else "Unknown Assignment"
        
        # Parse sections
        sections = parse_markdown_sections(md_content)
        
        assignment_num = extract_assignment_number(md_path.name)
        assignment_data = {
            'filename': f'ELEC576_Assignment_{assignment_num}.pdf',  # Match existing format
            'assignment_number': assignment_num,
            'title': assignment_title,
            'sections': sections
        }
        
        print(f"✓ {len(sections)} sections")
        return assignment_data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("🚀 Extracting sections from assignment markdown files...\n")
    
    # Paths
    input_dir = Path('data/raw/assignment')
    output_dir = Path('data/processed/assignments')
    
    if not input_dir.exists():
        print(f"❌ Assignment directory not found: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all markdown files
    md_files = sorted(input_dir.glob('assignment*.md'))
    
    if not md_files:
        print(f"❌ No markdown files found in {input_dir}")
        return
    
    print(f"📚 Found {len(md_files)} markdown files\n")
    
    all_assignments = []
    total_sections = 0
    
    # Process each markdown file
    for md_file in md_files:
        assignment_data = extract_text_from_markdown(md_file)
        
        if assignment_data:
            all_assignments.append(assignment_data)
            total_sections += len(assignment_data['sections'])
            
            # Save individual file (use ELEC576_Assignment_X format to match existing)
            assignment_num = assignment_data['assignment_number']
            output_file = output_dir / f"ELEC576_Assignment_{assignment_num}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(assignment_data, f, indent=2, ensure_ascii=False)
    
    # Save combined file
    combined_file = output_dir / 'all_assignments.json'
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_assignments': len(all_assignments),
            'total_sections': total_sections,
            'assignments': all_assignments
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Done!")
    print(f"{'='*60}")
    print(f"\n📊 Statistics:")
    print(f"   Assignments processed: {len(all_assignments)}")
    print(f"   Total sections: {total_sections}")
    print(f"   Avg sections per assignment: {total_sections / len(all_assignments):.1f}")
    
    print(f"\n💾 Output files:")
    print(f"   Individual files: {output_dir}/*.json")
    print(f"   Combined file: {combined_file}")
    
    # Show sample
    if all_assignments:
        print(f"\n📝 Sample assignment: {all_assignments[0]['filename']}")
        print(f"   Assignment #{all_assignments[0]['assignment_number']}")
        print(f"   Title: {all_assignments[0]['title']}")
        print(f"   Sections: {len(all_assignments[0]['sections'])}")
        if all_assignments[0]['sections']:
            print(f"\n   Section examples:")
            for i, section in enumerate(all_assignments[0]['sections'][:3], 1):
                print(f"     {i}. {section['title'][:60]}")
                print(f"        Content: {len(section['content'])} chars")


if __name__ == '__main__':
    main()

