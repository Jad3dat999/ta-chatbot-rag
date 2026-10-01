#!/usr/bin/env python3
"""
Improved Assignment Extraction - Groups by major tasks

Structure:
- Overview (due date, submission, resources)
- Problem 1 (ALL sub-parts a, b, c, 1, 2, 3 together)
- Problem 2 (ALL sub-parts together)
- Policies (collaboration, plagiarism)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional

def clean_text(text: str) -> str:
    """Clean text while preserving structure"""
    if not text:
        return ''
    text = re.sub(r'\n\n\n+', '\n\n', text)  # Max 2 newlines
    return text.strip()


def extract_assignment_number(filename: str) -> str:
    """Extract assignment number"""
    match = re.search(r'[Aa]ssignment[_\s]*(\d+)', filename)
    return match.group(1) if match else 'unknown'


def parse_markdown_assignment(text: str) -> List[Dict]:
    """
    Parse markdown assignment into major sections
    
    Rules:
    - `# 1. Task Name` = Major task (group ALL content until next # 1. or # 2.)
    - `## Submission` = Policy section
    - Everything before first `# 1.` = Overview
    """
    sections = []
    lines = text.split('\n')
    
    # Find major boundaries
    boundaries = []
    
    for i, line in enumerate(lines):
        # Major task: # 1. or # 2. (but not ## )
        if re.match(r'^#\s+(\d+)\.\s+(.+)$', line.strip()):
            match = re.match(r'^#\s+(\d+)\.\s+(.+)$', line.strip())
            boundaries.append({
                'line': i,
                'type': 'problem',
                'number': match.group(1),
                'title': match.group(2).strip()
            })
        
        # Policy sections: ## Submission, ## Collaboration, ## Plagiarism, ## LLM
        elif re.match(r'^##\s+(Submission|Collaboration|Plagiarism|LLM|GPU)\s', line.strip(), re.IGNORECASE):
            match = re.match(r'^##\s+(.+?)(?:\s+Instructions?|\s+Policy|\s+Resource)?$', line.strip(), re.IGNORECASE)
            title = match.group(1).strip() if match else line.strip('# ').strip()
            boundaries.append({
                'line': i,
                'type': 'policy',
                'number': None,
                'title': title
            })
    
    if not boundaries:
        # No structure found, return whole thing
        return [{'title': 'Full Assignment', 'content': clean_text(text)}]
    
    # Extract sections
    # Overview: everything before first boundary
    first_boundary = boundaries[0]
    overview_lines = lines[:first_boundary['line']]
    overview_text = '\n'.join(overview_lines).strip()
    
    if overview_text and len(overview_text) > 30:
        sections.append({
            'title': 'Assignment Overview and Requirements',
            'content': clean_text(overview_text)
        })
    
    # Extract each major section
    for i, boundary in enumerate(boundaries):
        start_line = boundary['line']
        
        # Find end line (next boundary or end of file)
        if i + 1 < len(boundaries):
            end_line = boundaries[i + 1]['line']
        else:
            end_line = len(lines)
        
        # Get section content (including the title line)
        section_lines = lines[start_line:end_line]
        section_text = '\n'.join(section_lines).strip()
        
        # Create title
        if boundary['type'] == 'problem':
            title = f"Problem {boundary['number']}: {boundary['title']}"
        else:
            title = boundary['title']
        
        if section_text and len(section_text) > 30:
            sections.append({
                'title': title,
                'content': clean_text(section_text)
            })
    
    return sections


def extract_from_markdown(md_path: Path) -> Optional[Dict]:
    """Extract from markdown file"""
    print(f"  {md_path.name}...", end=' ')
    
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        sections = parse_markdown_assignment(text)
        
        print(f"✓ {len(sections)} sections")
        
        return {
            'filename': md_path.stem + '.pdf',
            'assignment_number': extract_assignment_number(md_path.name),
            'title': f"Assignment {extract_assignment_number(md_path.name)}",
            'sections': sections
        }
        
    except Exception as e:
        print(f"❌ {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("🚀 Extracting assignments with improved grouping\n")
    
    input_dir = Path('data/raw/assignment')
    output_dir = Path('data/processed/assignments')
    
    if not input_dir.exists():
        print(f"❌ Not found: {input_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find markdown files
    md_files = sorted(input_dir.glob('assignment*.md'))
    
    if not md_files:
        print(f"❌ No assignment*.md files found in {input_dir}")
        print(f"   Looking for files like: assignment0.md, assignment1.md, ...")
        return
    
    print(f"📚 Found {len(md_files)} assignments\n")
    
    all_assignments = []
    
    for md_file in md_files:
        assignment_data = extract_from_markdown(md_file)
        
        if assignment_data:
            all_assignments.append(assignment_data)
            
            # Save individual JSON
            output_file = output_dir / f"{md_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(assignment_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Processed {len(all_assignments)} assignments")
    print(f"{'='*60}\n")
    
    # Show structure
    for assignment in all_assignments:
        print(f"📝 {assignment['title']} ({assignment['filename']})")
        print(f"   Sections: {len(assignment['sections'])}")
        for section in assignment['sections']:
            content_len = len(section['content'])
            print(f"      • {section['title']:<50} ({content_len:,} chars)")
        print()
    
    print(f"💾 Saved to: {output_dir}/")


if __name__ == '__main__':
    main()
