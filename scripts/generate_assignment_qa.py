#!/usr/bin/env python3
"""
Generate synthetic Q&A pairs from assignment documents for retriever training.

This script creates diverse question-answer pairs from assignment sections to help
the retriever learn to retrieve assignment documents when asked assignment-specific questions.
"""

import json
from pathlib import Path
from typing import List, Dict
import re


def generate_qa_for_assignment(assignment: Dict) -> List[Dict]:
    """
    Generate multiple Q&A pairs for a single assignment
    
    Args:
        assignment: Assignment data with sections
        
    Returns:
        List of Q&A pairs (dicts with 'question' and 'answer' keys)
    """
    qa_pairs = []
    
    assign_num = assignment.get('assignment_number', '?')
    assign_title = assignment.get('title', f'Assignment {assign_num}')
    sections = assignment.get('sections', [])
    
    # Find key sections
    overview_section = None
    submission_section = None
    problem_sections = []
    
    for section in sections:
        title = section.get('title', '').lower()
        if 'overview' in title or 'requirement' in title:
            overview_section = section
        elif 'submission' in title:
            submission_section = section
        elif 'problem' in title or 'task' in title:
            problem_sections.append(section)
    
    # ========== Generate Overview/Requirements Q&A ==========
    if overview_section:
        content = overview_section['content']
        
        # Q1: General requirements
        qa_pairs.append({
            'question': f"What are the requirements for assignment {assign_num}?",
            'answer': content,
            'metadata': {
                'type': 'assignment_requirements',
                'assignment': assign_num,
                'source': 'synthetic'
            }
        })
        
        # Q2: Alternative phrasing
        qa_pairs.append({
            'question': f"What do I need to do for assignment {assign_num}?",
            'answer': content,
            'metadata': {
                'type': 'assignment_requirements',
                'assignment': assign_num,
                'source': 'synthetic'
            }
        })
        
        # Q3: Due date question (if due date is in content)
        if 'due' in content.lower():
            qa_pairs.append({
                'question': f"When is assignment {assign_num} due?",
                'answer': content,
                'metadata': {
                    'type': 'assignment_due_date',
                    'assignment': assign_num,
                    'source': 'synthetic'
                }
            })
            
            qa_pairs.append({
                'question': f"What is the due date for assignment {assign_num}?",
                'answer': content,
                'metadata': {
                    'type': 'assignment_due_date',
                    'assignment': assign_num,
                    'source': 'synthetic'
                }
            })
    
    # ========== Generate Submission Q&A ==========
    if submission_section:
        content = submission_section['content']
        
        qa_pairs.append({
            'question': f"How do I submit assignment {assign_num}?",
            'answer': content,
            'metadata': {
                'type': 'assignment_submission',
                'assignment': assign_num,
                'source': 'synthetic'
            }
        })
        
        qa_pairs.append({
            'question': f"What are the submission instructions for assignment {assign_num}?",
            'answer': content,
            'metadata': {
                'type': 'assignment_submission',
                'assignment': assign_num,
                'source': 'synthetic'
            }
        })
    
    # ========== Generate Problem-Specific Q&A ==========
    for problem_section in problem_sections:
        section_title = problem_section.get('title', '')
        content = problem_section['content']
        
        # Extract problem number (e.g., "Problem 1: Backpropagation..." -> "1")
        problem_match = re.search(r'problem\s+(\d+)', section_title.lower())
        if problem_match:
            problem_num = problem_match.group(1)
            
            # Extract problem description (first line or title after "Problem N:")
            problem_desc_match = re.search(r'problem\s+\d+:\s*(.+?)(?:\n|$)', section_title, re.IGNORECASE)
            problem_desc = problem_desc_match.group(1) if problem_desc_match else ""
            
            # Q1: What is problem X about?
            qa_pairs.append({
                'question': f"What is problem {problem_num} in assignment {assign_num}?",
                'answer': content,
                'metadata': {
                    'type': 'assignment_problem',
                    'assignment': assign_num,
                    'problem': problem_num,
                    'source': 'synthetic'
                }
            })
            
            # Q2: What do I need to implement for problem X?
            qa_pairs.append({
                'question': f"What do I need to implement for problem {problem_num} in assignment {assign_num}?",
                'answer': content,
                'metadata': {
                    'type': 'assignment_problem',
                    'assignment': assign_num,
                    'problem': problem_num,
                    'source': 'synthetic'
                }
            })
            
            # Q3: Explain problem X (if there's a description)
            if problem_desc:
                qa_pairs.append({
                    'question': f"Explain the {problem_desc.lower()} problem in assignment {assign_num}",
                    'answer': content,
                    'metadata': {
                        'type': 'assignment_problem',
                        'assignment': assign_num,
                        'problem': problem_num,
                        'source': 'synthetic'
                    }
                })
                
                # Q4: How to solve [problem description]?
                qa_pairs.append({
                    'question': f"How do I solve the {problem_desc.lower()} in assignment {assign_num}?",
                    'answer': content,
                    'metadata': {
                        'type': 'assignment_problem',
                        'assignment': assign_num,
                        'problem': problem_num,
                        'source': 'synthetic'
                    }
                })
            
            # Q5: Requirements for problem X
            qa_pairs.append({
                'question': f"What are the requirements for problem {problem_num} in assignment {assign_num}?",
                'answer': content,
                'metadata': {
                    'type': 'assignment_problem',
                    'assignment': assign_num,
                    'problem': problem_num,
                    'source': 'synthetic'
                }
            })
    
    return qa_pairs


def main():
    """Generate Q&A pairs from all assignments"""
    
    # Paths
    assignments_dir = Path('data/processed/assignments')
    output_file = Path('data/processed/assignment_qa_pairs.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("🔧 Generating synthetic Q&A pairs from assignments...")
    print(f"   Reading from: {assignments_dir}")
    
    all_qa_pairs = []
    
    # Process each assignment file
    assignment_files = sorted(assignments_dir.glob('assignment*.json'))
    
    for assign_file in assignment_files:
        print(f"\n📄 Processing {assign_file.name}...")
        
        with open(assign_file, 'r', encoding='utf-8') as f:
            assignment = json.load(f)
        
        # Generate Q&A pairs
        qa_pairs = generate_qa_for_assignment(assignment)
        all_qa_pairs.extend(qa_pairs)
        
        print(f"   Generated {len(qa_pairs)} Q&A pairs")
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_qa_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Generated {len(all_qa_pairs)} total Q&A pairs")
    print(f"   Saved to: {output_file}")
    
    # Print some examples
    print("\n📝 Example Q&A pairs:")
    print("=" * 80)
    for i, qa in enumerate(all_qa_pairs[:3], 1):
        print(f"\n{i}. Question: {qa['question']}")
        print(f"   Answer preview: {qa['answer'][:150]}...")
        print(f"   Metadata: {qa['metadata']}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()





