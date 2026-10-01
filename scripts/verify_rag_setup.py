#!/usr/bin/env python3
"""
Verify RAG setup is ready for training
"""

import json
from pathlib import Path


def verify_data_files():
    """Verify all required data files exist"""
    print("🔍 Verifying data files...\n")
    
    required_files = {
        'Piazza Q&A': 'data/processed/piazza_qa.json',
        'All Data': 'data/processed/all_data.json',
        'Train Split': 'data/splits/train.json',
        'Val Split': 'data/splits/val.json',
        'Test Split': 'data/splits/test.json',
        'Documents': 'data/splits/documents.json'
    }
    
    all_good = True
    for name, path in required_files.items():
        file_path = Path(path)
        if file_path.exists():
            size = file_path.stat().st_size / 1024  # KB
            print(f"  ✅ {name}: {path} ({size:.1f} KB)")
        else:
            print(f"  ❌ {name}: {path} - NOT FOUND")
            all_good = False
    
    return all_good


def verify_data_structure():
    """Verify data structure is correct"""
    print("\n🔍 Verifying data structure...\n")
    
    try:
        # Check train data
        with open('data/splits/train.json', 'r') as f:
            train_data = json.load(f)
        
        if not train_data:
            print("  ❌ Train data is empty")
            return False
        
        sample = train_data[0]
        required_fields = ['question', 'answer', 'source']
        missing = [f for f in required_fields if f not in sample]
        
        if missing:
            print(f"  ❌ Train data missing fields: {missing}")
            return False
        
        print(f"  ✅ Train data: {len(train_data)} examples")
        print(f"     Sample question: {sample['question'][:60]}...")
        
        # Check documents
        with open('data/splits/documents.json', 'r') as f:
            docs = json.load(f)
        
        if not docs:
            print("  ❌ Documents are empty")
            return False
        
        sample_doc = docs[0]
        required_doc_fields = ['text', 'source', 'doc_id', 'doc_type']
        missing_doc = [f for f in required_doc_fields if f not in sample_doc]
        
        if missing_doc:
            print(f"  ❌ Documents missing fields: {missing_doc}")
            return False
        
        # Count by type
        slide_count = len([d for d in docs if d['doc_type'] == 'slide'])
        assign_count = len([d for d in docs if d['doc_type'] == 'assignment'])
        piazza_note_count = len([d for d in docs if d['doc_type'] == 'piazza_note'])
        piazza_qa_count = len([d for d in docs if d['doc_type'] == 'piazza_qa'])
        piazza_count = piazza_note_count + piazza_qa_count
        
        print(f"  ✅ Documents: {len(docs)} total")
        print(f"     - Slides: {slide_count}")
        print(f"     - Assignments: {assign_count}")
        print(f"     - Piazza: {piazza_count} (notes: {piazza_note_count}, Q&A: {piazza_qa_count})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error verifying structure: {e}")
        return False


def verify_code_files():
    """Verify RAG code files exist"""
    print("\n🔍 Verifying code files...\n")
    
    required_files = {
        'Retriever': 'src/retriever.py',
        'Generator': 'src/generator.py',
        'Chatbot': 'src/chatbot.py',
        'Evaluation': 'src/evaluation.py',
        'Data Prep': 'scripts/prepare_rag_data.py'
    }
    
    all_good = True
    for name, path in required_files.items():
        file_path = Path(path)
        if file_path.exists():
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: {path} - NOT FOUND")
            all_good = False
    
    return all_good


def main():
    print("="*60)
    print("RAG Setup Verification")
    print("="*60 + "\n")
    
    checks = [
        ("Data Files", verify_data_files),
        ("Data Structure", verify_data_structure),
        ("Code Files", verify_code_files)
    ]
    
    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All checks passed! Ready for RAG training.")
        print("\nNext steps:")
        print("  1. Upload project to Colab")
        print("  2. Train retriever (Sentence-BERT)")
        print("  3. Train generator (Mistral-7B LoRA)")
        print("  4. Build RAG pipeline")
    else:
        print("❌ Some checks failed. Please fix issues before training.")
    print("="*60)


if __name__ == '__main__':
    main()



