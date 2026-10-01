"""
Utility functions for document handling
"""

from typing import List, Dict


def is_piazza_document(doc: Dict) -> bool:
    """Check if a document is a Piazza document (note or Q&A)"""
    return doc.get('doc_type') in ['piazza_note', 'piazza_qa']


def count_documents_by_type(docs: List[Dict]) -> Dict[str, int]:
    """
    Count documents by type
    
    Returns:
        Dictionary with counts for each type and subtypes
    """
    counts = {
        'slide': 0,
        'assignment': 0,
        'piazza_note': 0,
        'piazza_qa': 0,
        'piazza_total': 0,
        'total': len(docs)
    }
    
    for doc in docs:
        doc_type = doc.get('doc_type', 'unknown')
        
        if doc_type == 'slide':
            counts['slide'] += 1
        elif doc_type == 'assignment':
            counts['assignment'] += 1
        elif doc_type == 'piazza_note':
            counts['piazza_note'] += 1
            counts['piazza_total'] += 1
        elif doc_type == 'piazza_qa':
            counts['piazza_qa'] += 1
            counts['piazza_total'] += 1
    
    return counts


def print_document_stats(docs: List[Dict], title: str = "Document Statistics"):
    """Print formatted document statistics"""
    counts = count_documents_by_type(docs)
    
    print(f"\n📊 {title}")
    print(f"   Total: {counts['total']} documents")
    print(f"   - Slides: {counts['slide']}")
    print(f"   - Assignments: {counts['assignment']}")
    print(f"   - Piazza: {counts['piazza_total']} (notes: {counts['piazza_note']}, Q&A: {counts['piazza_qa']})")


def filter_documents_for_training(docs: List[Dict]) -> List[Dict]:
    """
    Filter documents for training (only piazza_qa, not piazza_note)
    
    For retrieval training, we only use Q&A pairs from Piazza,
    not the instructor notes.
    """
    return [d for d in docs if d.get('doc_type') != 'piazza_note']


def filter_documents_for_retrieval(docs: List[Dict]) -> List[Dict]:
    """
    Filter documents for retrieval (includes everything)
    
    For retrieval/indexing, we use ALL document types:
    - Slides
    - Assignments
    - Piazza notes
    - Piazza Q&A
    """
    return docs  # Use all documents



