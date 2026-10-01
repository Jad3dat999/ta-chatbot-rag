#!/usr/bin/env python3
"""
Index documents for retrieval using fine-tuned retriever
Run this after training the retriever

Usage:
    python scripts/index_documents.py --model_path models/retriever-finetuned
"""

import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever import Retriever
from src.chatbot import ChromaDBManager


def load_documents(documents_file: Path):
    """Load documents for indexing"""
    print(f"📖 Loading documents from {documents_file}...")
    
    with open(documents_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"   Loaded {len(documents)} documents")
    
    # Extract text and metadata (including tags and semantic hints)
    texts = [doc['text'] for doc in documents]
    metadatas = [
        {
            'source': doc.get('source', 'unknown'),
            'source_detail': doc.get('source_detail', ''),
            'doc_type': doc.get('doc_type', 'unknown'),
            'doc_id': doc.get('doc_id', ''),
            'tags': doc.get('tags', []),
            # Semantic hints for better retrieval ranking
            'is_definition': doc.get('is_definition', False),
            'contains_equation': doc.get('contains_equation', False),
            'importance': doc.get('importance', 'medium')
        }
        for doc in documents
    ]
    ids = [doc.get('doc_id', f'doc_{i}') for i, doc in enumerate(documents)]
    
    return texts, metadatas, ids


def main():
    parser = argparse.ArgumentParser(description='Index documents for retrieval')
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/retriever-finetuned',
        help='Path to fine-tuned retriever model'
    )
    parser.add_argument(
        '--documents_file',
        type=str,
        default='data/splits/documents.json',
        help='Path to documents JSON file'
    )
    parser.add_argument(
        '--use_chromadb',
        action='store_true',
        help='Also index in ChromaDB (in addition to Sentence-BERT)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔍 Indexing Documents for Retrieval")
    print("="*60 + "\n")
    
    model_path = Path(args.model_path)
    documents_file = Path(args.documents_file)
    
    if not model_path.exists():
        print(f"❌ Error: Model path not found: {model_path}")
        print("   Please train the retriever first using train_retriever.py")
        return
    
    if not documents_file.exists():
        print(f"❌ Error: Documents file not found: {documents_file}")
        return
    
    # Load documents
    texts, metadatas, ids = load_documents(documents_file)
    
    # Initialize retriever
    print(f"\n🔧 Loading retriever from {model_path}...")
    retriever = Retriever(str(model_path))
    
    # Index documents with Sentence-BERT
    print("\n📚 Indexing documents with Sentence-BERT...")
    retriever.index_documents(texts)
    
    # Save document metadata for later use
    metadata_file = model_path / 'document_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            'documents': [
                {
                    'id': doc_id,
                    'text': text[:100] + '...' if len(text) > 100 else text,  # Preview only
                    'metadata': meta
                }
                for text, meta, doc_id in zip(texts, metadatas, ids)
            ],
            'total_documents': len(texts)
        }, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ Document metadata saved to {metadata_file}")
    
    # Optionally index in ChromaDB
    if args.use_chromadb:
        print("\n💾 Indexing in ChromaDB...")
        db_manager = ChromaDBManager(persist_directory="data/chromadb")
        db_manager.create_collection("ta_chatbot_docs")
        db_manager.add_documents(texts, metadatas, ids)
        print("   ✓ Documents indexed in ChromaDB")
    
    # Test retrieval
    print("\n🧪 Testing retrieval...")
    test_query = "How does backpropagation work?"
    results = retriever.retrieve(test_query, top_k=3)
    
    print(f"\n   Query: '{test_query}'")
    print(f"   Retrieved {len(results)} documents:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n   {i}. Score: {score:.4f}")
        print(f"      Preview: {doc[:150]}...")
    
    print("\n" + "="*60)
    print("✅ Document indexing complete!")
    print("="*60)
    print(f"\n💾 Retriever ready with {len(texts)} indexed documents")
    print("\n📝 Next steps:")
    print("   1. Use this retriever to get context for generator training")
    print("   2. Or use it directly in the RAG pipeline")


if __name__ == '__main__':
    main()



if __name__ == '__main__':
    main()

