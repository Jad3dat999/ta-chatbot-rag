#!/usr/bin/env python3
"""
Prepare data for RAG training - UPDATED for new Piazza data format
- Combines Piazza Q&A, Piazza notes, slides, and assignments
- Creates documents.json with 4 types: slides, assignments, piazza_note, piazza_qa
"""

import json
import random
from pathlib import Path
from typing import List, Dict


def load_all_data(filepath: str) -> Dict:
    """Load all data (lectures + assignments)"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_new_piazza_qa(filepath: str) -> List[Dict]:
    """Load NEW hierarchical Piazza Q&A data"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_piazza_notes(filepath: str) -> List[Dict]:
    """Load Piazza notes"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_slides_for_retrieval(slides_data: List[Dict]) -> List[Dict]:
    """Prepare slides as retrieval documents"""
    documents = []
    
    for lecture in slides_data:
        lecture_title = lecture.get('title', 'Unknown')
        lecture_num = lecture.get('lecture_number', 'unknown')
        filename = lecture.get('filename', 'unknown')
        
        for slide in lecture.get('slides', []):
            slide_text = slide.get('text', '').strip()
            # Skip slides with minimal content
            if len(slide_text) < 20:
                continue
                
            documents.append({
                'text': slide_text,
                'source': f"{lecture_title} - Slide {slide['page']}",
                'source_detail': f"{filename}, Page {slide['page']}",
                'lecture': lecture_title,
                'lecture_number': lecture_num,
                'page': slide['page'],
                'content_type': slide.get('content_type', 'text'),
                'tags': slide.get('tags', []),
                'doc_id': f"slide_{lecture_num}_{slide['page']}",
                'doc_type': 'slide'
            })
    
    return documents


def prepare_assignments_for_retrieval(assignments_data: List[Dict]) -> List[Dict]:
    """Prepare assignment sections as retrieval documents"""
    documents = []
    
    for assignment in assignments_data:
        assign_title = assignment.get('title', 'Unknown Assignment')
        assign_num = assignment.get('assignment_number', 'unknown')
        filename = assignment.get('filename', 'unknown')
        
        for section in assignment.get('sections', []):
            section_text = section.get('content', '').strip()
            # Skip sections with minimal content
            if len(section_text) < 20:
                continue
                
            section_title = section.get('title', 'General')
            
            documents.append({
                'text': section_text,
                'source': f"{assign_title} - {section_title}",
                'source_detail': f"{filename}, Section: {section_title}",
                'assignment': assign_title,
                'assignment_number': assign_num,
                'section_title': section_title,
                'doc_id': f"assignment_{assign_num}_{section_title.replace(' ', '_')[:30]}",
                'doc_type': 'assignment'
            })
    
    return documents


def prepare_piazza_notes_for_retrieval(notes_data: List[Dict]) -> List[Dict]:
    """Prepare Piazza notes as retrieval documents"""
    documents = []
    
    for i, note in enumerate(notes_data):
        subject = note.get('subject', 'Untitled Note')
        content = note.get('content', '').strip()
        
        # Skip if no content
        if len(content) < 20:
            continue
        
        # Combine subject and content for better retrieval
        text = f"{subject}\n{content}"
        
        documents.append({
            'text': text,
            'source': f"Piazza Note: {subject}",
            'source_detail': f"Piazza Instructor Note",
            'subject': subject,
            'tags': note.get('tags', []),
            'doc_id': f"piazza_note_{i}",
            'doc_type': 'piazza_note'
        })
    
    return documents


def prepare_piazza_qa_for_retrieval(qa_threads: List[Dict]) -> List[Dict]:
    """
    Prepare Piazza Q&A as retrieval documents
    
    For each thread:
    - Create doc from each main answer
    - Create doc from each followup answer
    """
    documents = []
    doc_id = 1
    
    for thread in qa_threads:
        main_question = thread.get('question', '')
        main_answers = thread.get('answers', [])
        followups = thread.get('followups', [])
        tags = thread.get('tags', [])
        
        # Create documents from main answers
        for answer in main_answers:
            if len(answer.strip()) < 20:
                continue
            
            # Include question context in the document for better retrieval
            text = f"Q: {main_question}\n\nA: {answer}"
            
            documents.append({
                'text': text,
                'source': f"Piazza Q&A",
                'source_detail': f"Piazza Question & Answer",
                'question': main_question,
                'answer': answer,
                'tags': tags,
                'doc_id': f"piazza_qa_{doc_id}",
                'doc_type': 'piazza_qa',
                'is_followup': False
            })
            doc_id += 1
        
        # Create documents from followup answers
        for followup in followups:
            followup_question = followup.get('question', '')
            followup_answers = followup.get('answers', [])
            
            for answer in followup_answers:
                if len(answer.strip()) < 20:
                    continue
                
                # Include both parent and followup question for context
                text = f"Parent Q: {main_question}\n\nFollowup Q: {followup_question}\n\nA: {answer}"
                
                documents.append({
                    'text': text,
                    'source': f"Piazza Q&A (Followup)",
                    'source_detail': f"Piazza Followup Question & Answer",
                    'parent_question': main_question,
                    'question': followup_question,
                    'answer': answer,
                    'tags': tags,
                    'doc_id': f"piazza_qa_{doc_id}",
                    'doc_type': 'piazza_qa',
                    'is_followup': True
                })
                doc_id += 1
    
    return documents


def main():
    print("="*60)
    print("🚀 Preparing RAG Retrieval Documents")
    print("="*60)
    
    # Paths
    all_data_file = Path('data/processed/all_data.json')
    piazza_qa_file = Path('data/processed/piazza_qa_pairs.json')
    piazza_notes_file = Path('data/processed/piazza_notes.json')
    output_dir = Path('data/splits')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check files exist
    if not all_data_file.exists():
        print(f"❌ Error: {all_data_file} not found!")
        return
    
    if not piazza_qa_file.exists():
        print(f"❌ Error: {piazza_qa_file} not found!")
        return
    
    if not piazza_notes_file.exists():
        print(f"❌ Error: {piazza_notes_file} not found!")
        return
    
    # Load data
    print("\n📖 Loading data...")
    all_data = load_all_data(all_data_file)
    piazza_qa_threads = load_new_piazza_qa(piazza_qa_file)
    piazza_notes = load_piazza_notes(piazza_notes_file)
    
    lectures = all_data.get('lectures', [])
    assignments = all_data.get('assignments', [])
    
    print(f"   Lectures: {len(lectures)}")
    print(f"   Assignments: {len(assignments)}")
    print(f"   Piazza Q&A threads: {len(piazza_qa_threads)}")
    print(f"   Piazza notes: {len(piazza_notes)}")
    
    # Process data into retrieval documents
    print("\n🔄 Creating retrieval documents...")
    
    slide_docs = prepare_slides_for_retrieval(lectures)
    print(f"   ✓ Slides: {len(slide_docs)} documents")
    
    assignment_docs = prepare_assignments_for_retrieval(assignments)
    print(f"   ✓ Assignments: {len(assignment_docs)} documents")
    
    piazza_note_docs = prepare_piazza_notes_for_retrieval(piazza_notes)
    print(f"   ✓ Piazza notes: {len(piazza_note_docs)} documents")
    
    piazza_qa_docs = prepare_piazza_qa_for_retrieval(piazza_qa_threads)
    print(f"   ✓ Piazza Q&A: {len(piazza_qa_docs)} documents")
    
    # Combine all documents
    all_documents = slide_docs + assignment_docs + piazza_note_docs + piazza_qa_docs
    
    print(f"\n📊 Total documents: {len(all_documents)}")
    print(f"   - Slides: {len(slide_docs)}")
    print(f"   - Assignments: {len(assignment_docs)}")
    print(f"   - Piazza notes: {len(piazza_note_docs)}")
    print(f"   - Piazza Q&A: {len(piazza_qa_docs)}")
    
    # Save documents
    documents_file = output_dir / 'documents.json'
    print(f"\n💾 Saving to {documents_file}...")
    
    with open(documents_file, 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ Saved {len(all_documents)} documents")
    
    # Verify document types
    print("\n✅ Document types verified:")
    doc_types = {}
    for doc in all_documents:
        doc_type = doc['doc_type']
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    for doc_type, count in sorted(doc_types.items()):
        print(f"   - {doc_type}: {count}")
    
    # Show samples
    print("\n📄 Sample documents:")
    for doc_type in ['slide', 'assignment', 'piazza_note', 'piazza_qa']:
        sample = next((d for d in all_documents if d['doc_type'] == doc_type), None)
        if sample:
            print(f"\n   {doc_type}:")
            print(f"      Source: {sample['source']}")
            print(f"      Text: {sample['text'][:80]}...")
    
    print("\n" + "="*60)
    print("✅ Documents ready for retrieval!")
    print("="*60)
    print("\n🎯 Next step: Run training notebook to index these documents")


if __name__ == '__main__':
    main()
