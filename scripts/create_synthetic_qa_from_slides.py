#!/usr/bin/env python3
"""
Optional: Create synthetic Q&A pairs from lecture slides and assignments
to augment the training data for the retriever.

This script shows how to use all_data.json (slides + assignments) to
create additional training examples beyond the Piazza Q&A pairs.

Usage:
    python scripts/create_synthetic_qa_from_slides.py
"""

import json
from pathlib import Path
from typing import List, Dict
import random


def create_qa_from_slide(slide: Dict) -> List[Dict]:
    """
    Create Q&A pairs from a slide.
    
    Simple heuristic approach:
    - Use slide text as "context"
    - Generate simple questions like "What is covered in [lecture]?"
    
    For production use, you could:
    - Use an LLM to generate better questions
    - Extract key concepts and create targeted questions
    - Use the topics field to generate topic-specific questions
    
    Args:
        slide: Slide dictionary from all_data.json
        
    Returns:
        List of Q&A dictionaries
    """
    qa_pairs = []
    
    text = slide.get('text', '').strip()
    if not text or len(text) < 50:  # Skip very short slides
        return qa_pairs
    
    page = slide.get('page', 'unknown')
    has_equations = slide.get('has_equations', False)
    has_figures = slide.get('has_figures', False)
    topics = slide.get('topics', [])
    
    # Simple question templates
    questions = []
    
    # Generic question about the slide content
    questions.append(f"What is discussed on slide {page}?")
    
    # Topic-based questions
    if topics:
        for topic in topics:
            questions.append(f"What information is provided about {topic}?")
    
    # Content-type specific questions
    if has_equations:
        questions.append(f"What mathematical concepts are covered on slide {page}?")
    if has_figures:
        questions.append(f"What diagrams or figures are shown on slide {page}?")
    
    # Create Q&A pairs
    for question in questions:
        qa_pairs.append({
            'question': question,
            'answer': text,
            'source': f"slides_synthetic",
            'source_detail': f"Slide {page} - Synthetic Q&A"
        })
    
    return qa_pairs


def create_qa_from_assignment(section: Dict, assignment_name: str) -> List[Dict]:
    """
    Create Q&A pairs from an assignment section.
    
    Args:
        section: Assignment section dictionary
        assignment_name: Name of the assignment
        
    Returns:
        List of Q&A dictionaries
    """
    qa_pairs = []
    
    title = section.get('title', '').strip()
    content = section.get('content', '').strip()
    
    if not content or len(content) < 50:
        return qa_pairs
    
    # Simple questions about assignment requirements
    questions = [
        f"What are the requirements for {title}?",
        f"What is {title} in {assignment_name}?",
        f"How do I complete {title}?",
    ]
    
    for question in questions:
        qa_pairs.append({
            'question': question,
            'answer': content,
            'source': f"assignments_synthetic",
            'source_detail': f"{assignment_name} - {title}"
        })
    
    return qa_pairs


def main():
    print("="*60)
    print("📚 Creating Synthetic Q&A from Slides and Assignments")
    print("="*60 + "\n")
    
    # Load all_data.json
    all_data_path = Path('data/processed/all_data.json')
    if not all_data_path.exists():
        print(f"❌ Error: {all_data_path} not found")
        return
    
    print(f"📖 Loading {all_data_path}...")
    with open(all_data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    metadata = all_data.get('metadata', {})
    print(f"   Total lectures: {metadata.get('total_lectures', 0)}")
    print(f"   Total slides: {metadata.get('total_slides', 0)}")
    print(f"   Total assignments: {metadata.get('total_assignments', 0)}")
    print(f"   Total assignment sections: {metadata.get('total_assignment_sections', 0)}")
    
    # Generate Q&A from slides
    print(f"\n📄 Generating Q&A from slides...")
    slide_qa_pairs = []
    
    for lecture in all_data.get('lectures', []):
        slides = lecture.get('slides', [])
        for slide in slides:
            qa_pairs = create_qa_from_slide(slide)
            slide_qa_pairs.extend(qa_pairs)
    
    print(f"   Generated {len(slide_qa_pairs)} Q&A pairs from slides")
    
    # Generate Q&A from assignments
    print(f"\n📝 Generating Q&A from assignments...")
    assignment_qa_pairs = []
    
    for assignment in all_data.get('assignments', []):
        assignment_name = assignment.get('filename', 'Unknown Assignment')
        sections = assignment.get('sections', [])
        for section in sections:
            qa_pairs = create_qa_from_assignment(section, assignment_name)
            assignment_qa_pairs.extend(qa_pairs)
    
    print(f"   Generated {len(assignment_qa_pairs)} Q&A pairs from assignments")
    
    # Combine all synthetic Q&A
    all_synthetic_qa = slide_qa_pairs + assignment_qa_pairs
    print(f"\n📊 Total synthetic Q&A pairs: {len(all_synthetic_qa)}")
    
    # Load existing training data
    train_path = Path('data/splits/train.json')
    if train_path.exists():
        with open(train_path, 'r', encoding='utf-8') as f:
            existing_train = json.load(f)
        print(f"   Existing training examples: {len(existing_train)}")
    else:
        existing_train = []
        print(f"   No existing training data found")
    
    # Sample synthetic data to avoid overwhelming the dataset
    # (You can adjust this ratio or remove sampling to use all synthetic data)
    max_synthetic_samples = min(len(all_synthetic_qa), len(existing_train) * 2)
    sampled_synthetic = random.sample(all_synthetic_qa, max_synthetic_samples) if len(all_synthetic_qa) > max_synthetic_samples else all_synthetic_qa
    
    print(f"   Sampled {len(sampled_synthetic)} synthetic examples (to balance with existing data)")
    
    # Combine and save
    combined_train = existing_train + sampled_synthetic
    output_path = Path('data/splits/train_with_synthetic.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_train, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved combined training data to: {output_path}")
    print(f"   Total examples: {len(combined_train)}")
    print(f"   - Original Q&A: {len(existing_train)}")
    print(f"   - Synthetic Q&A: {len(sampled_synthetic)}")
    
    # Show sample
    print(f"\n📄 Sample synthetic Q&A:")
    sample = random.choice(sampled_synthetic)
    print(f"   Question: {sample['question']}")
    print(f"   Answer (first 200 chars): {sample['answer'][:200]}...")
    print(f"   Source: {sample['source']}")
    
    print(f"\n{'='*60}")
    print(f"✅ Done!")
    print(f"{'='*60}")
    print(f"\nTo use this enhanced dataset for training:")
    print(f"1. Update your training script to load 'train_with_synthetic.json'")
    print(f"2. Or replace 'train.json' with this file")
    print(f"\nNote: The synthetic Q&A quality is basic. For production use,")
    print(f"consider using an LLM to generate higher-quality questions.")


if __name__ == '__main__':
    main()





