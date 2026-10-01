#!/usr/bin/env python3
"""
Combine all slides and assignments into a single comprehensive JSON file
"""

import json
from pathlib import Path
from typing import Dict, List


def load_slides(slides_dir: Path) -> List[Dict]:
    """Load all slide JSON files"""
    slides = []
    slide_files = sorted(slides_dir.glob('*.json'))
    
    print(f"📚 Loading {len(slide_files)} slide files...")
    for slide_file in slide_files:
        try:
            with open(slide_file, 'r', encoding='utf-8') as f:
                slide_data = json.load(f)
                slides.append(slide_data)
        except Exception as e:
            print(f"  ⚠️  Error loading {slide_file.name}: {e}")
    
    return slides


def load_assignments(assignments_dir: Path) -> List[Dict]:
    """Load all assignment JSON files"""
    assignments = []
    assignment_files = sorted(assignments_dir.glob('assignment*.json'))
    
    print(f"📝 Loading {len(assignment_files)} assignment files...")
    for assignment_file in assignment_files:
        try:
            with open(assignment_file, 'r', encoding='utf-8') as f:
                assignment_data = json.load(f)
                assignments.append(assignment_data)
        except Exception as e:
            print(f"  ⚠️  Error loading {assignment_file.name}: {e}")
    
    return assignments


def create_combined_json(slides: List[Dict], assignments: List[Dict], output_path: Path):
    """Create a combined JSON file with all slides and assignments"""
    
    combined_data = {
        'metadata': {
            'total_lectures': len(slides),
            'total_assignments': len(assignments),
            'total_slides': sum(len(lecture.get('slides', [])) for lecture in slides),
            'total_assignment_sections': sum(len(assignment.get('sections', [])) for assignment in assignments)
        },
        'lectures': slides,
        'assignments': assignments
    }
    
    # Save combined file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Combined JSON file created: {output_path}")
    print(f"   Total size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    print("🚀 Combining all slides and assignments into a single JSON file...\n")
    
    # Paths
    slides_dir = Path('data/processed/slides')
    assignments_dir = Path('data/processed/assignments')
    output_path = Path('data/processed/all_data.json')
    
    # Check directories exist
    if not slides_dir.exists():
        print(f"❌ Slides directory not found: {slides_dir}")
        return
    
    if not assignments_dir.exists():
        print(f"❌ Assignments directory not found: {assignments_dir}")
        return
    
    # Load data
    slides = load_slides(slides_dir)
    assignments = load_assignments(assignments_dir)
    
    if not slides and not assignments:
        print("❌ No data found to combine!")
        return
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create combined JSON
    create_combined_json(slides, assignments, output_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 Summary:")
    print(f"   Lectures: {len(slides)}")
    print(f"   Assignments: {len(assignments)}")
    print(f"   Total slides: {sum(len(lecture.get('slides', [])) for lecture in slides)}")
    print(f"   Total assignment sections: {sum(len(assignment.get('sections', [])) for assignment in assignments)}")
    print(f"{'='*60}")
    
    # Show sample structure
    if slides:
        print(f"\n📖 Sample lecture: {slides[0].get('filename', 'Unknown')}")
        print(f"   Title: {slides[0].get('title', 'N/A')}")
        print(f"   Slides: {len(slides[0].get('slides', []))}")
    
    if assignments:
        print(f"\n📝 Sample assignment: {assignments[0].get('filename', 'Unknown')}")
        print(f"   Title: {assignments[0].get('title', 'N/A')}")
        print(f"   Sections: {len(assignments[0].get('sections', []))}")


if __name__ == '__main__':
    main()

