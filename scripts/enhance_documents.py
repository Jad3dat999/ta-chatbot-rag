#!/usr/bin/env python3
"""
Enhance documents.json with LLM-powered metadata extraction:
1. Extract tags for documents with empty tags
2. Add semantic hints (is_definition, contains_equation, importance)
3. Chunk long documents (>1000 chars) with parent references

Supports OpenAI GPT-4 or Anthropic Claude APIs
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import time


def chunk_text(text: str, max_length: int = 800) -> List[str]:
    """
    Intelligently chunk text at sentence boundaries
    """
    # Simple sentence splitting
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]


def call_openai(text: str, api_key: str, task: str) -> Dict:
    """Call OpenAI GPT-4 API"""
    try:
        import openai
    except ImportError:
        print("❌ openai package not installed. Run: pip install openai")
        sys.exit(1)
    
    client = openai.OpenAI(api_key=api_key)
    
    if task == "tags":
        prompt = f"""Extract 2-5 relevant topics/keywords from this text. Return ONLY a JSON array of strings.
        
Text: {text[:500]}

Return format: ["topic1", "topic2", "topic3"]"""
    else:  # semantic hints
        prompt = f"""Analyze this text and return ONLY a JSON object with these fields:
- is_definition: true if it defines concepts (boolean)
- contains_equation: true if it has math equations (boolean)
- importance: "high", "medium", or "low" based on key concepts

Text: {text[:500]}

Return format: {{"is_definition": false, "contains_equation": true, "importance": "high"}}"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Fast and cheap
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100
    )
    
    result_text = response.choices[0].message.content.strip()
    # Remove markdown code blocks if present
    if result_text.startswith("```"):
        result_text = result_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    if result_text.startswith("json"):
        result_text = result_text[4:].strip()
    
    return json.loads(result_text)


def call_anthropic(text: str, api_key: str, task: str) -> Dict:
    """Call Anthropic Claude API"""
    try:
        import anthropic
    except ImportError:
        print("❌ anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)
    
    client = anthropic.Anthropic(api_key=api_key)
    
    if task == "tags":
        prompt = f"""Extract 2-5 relevant topics/keywords from this text. Return ONLY a JSON array of strings.
        
Text: {text[:500]}

Return format: ["topic1", "topic2", "topic3"]"""
    else:  # semantic hints
        prompt = f"""Analyze this text and return ONLY a JSON object with these fields:
- is_definition: true if it defines concepts (boolean)
- contains_equation: true if it has math equations (boolean)  
- importance: "high", "medium", or "low" based on key concepts

Text: {text[:500]}

Return format: {{"is_definition": false, "contains_equation": true, "importance": "high"}}"""
    
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",  # Fast and cheap
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    
    result_text = response.content[0].text.strip()
    # Remove markdown code blocks if present
    if result_text.startswith("```"):
        result_text = result_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    if result_text.startswith("json"):
        result_text = result_text[4:].strip()
    
    return json.loads(result_text)


def enhance_document(doc: Dict, api_key: str, provider: str, 
                    add_tags: bool = True, add_hints: bool = True) -> List[Dict]:
    """
    Enhance a single document with LLM-powered metadata
    Returns list of documents (1 if not chunked, multiple if chunked)
    """
    text = doc.get('text', '')
    
    # Decide if we need to chunk
    needs_chunking = len(text) > 1000
    
    if needs_chunking:
        # Chunk the document
        chunks = chunk_text(text, max_length=800)
        enhanced_docs = []
        
        for i, chunk_text in enumerate(chunks):
            chunk_doc = doc.copy()
            chunk_doc['text'] = chunk_text
            chunk_doc['parent_doc_id'] = doc['doc_id']
            chunk_doc['doc_id'] = f"{doc['doc_id']}_chunk{i+1}"
            chunk_doc['chunk_index'] = i + 1
            chunk_doc['total_chunks'] = len(chunks)
            
            # Add metadata for this chunk
            if add_tags and not chunk_doc.get('tags'):
                try:
                    if provider == "openai":
                        tags = call_openai(chunk_text, api_key, "tags")
                    else:
                        tags = call_anthropic(chunk_text, api_key, "tags")
                    chunk_doc['tags'] = tags if isinstance(tags, list) else []
                except Exception as e:
                    print(f"  ⚠️  Tag extraction failed: {e}")
                    chunk_doc['tags'] = []
            
            if add_hints:
                try:
                    if provider == "openai":
                        hints = call_openai(chunk_text, api_key, "hints")
                    else:
                        hints = call_anthropic(chunk_text, api_key, "hints")
                    chunk_doc.update(hints)
                except Exception as e:
                    print(f"  ⚠️  Hint extraction failed: {e}")
            
            enhanced_docs.append(chunk_doc)
        
        return enhanced_docs
    
    else:
        # Single document, no chunking needed
        enhanced_doc = doc.copy()
        
        # Add tags if empty
        if add_tags and not enhanced_doc.get('tags'):
            try:
                if provider == "openai":
                    tags = call_openai(text, api_key, "tags")
                else:
                    tags = call_anthropic(text, api_key, "tags")
                enhanced_doc['tags'] = tags if isinstance(tags, list) else []
            except Exception as e:
                print(f"  ⚠️  Tag extraction failed: {e}")
                enhanced_doc['tags'] = []
        
        # Add semantic hints
        if add_hints:
            try:
                if provider == "openai":
                    hints = call_openai(text, api_key, "hints")
                else:
                    hints = call_anthropic(text, api_key, "hints")
                enhanced_doc.update(hints)
            except Exception as e:
                print(f"  ⚠️  Hint extraction failed: {e}")
        
        return [enhanced_doc]


def main():
    print("="*60)
    print("🚀 Document Enhancement with LLM")
    print("="*60)
    
    # Try to load from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ Loaded .env file")
    except ImportError:
        print("ℹ️  python-dotenv not installed (optional)")
    
    # Get API provider choice
    print("\nChoose LLM provider:")
    print("  1. OpenAI GPT-4")
    print("  2. Anthropic Claude")
    provider_choice = input("Enter 1 or 2: ").strip()
    
    if provider_choice == "1":
        provider = "openai"
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            api_key = input("\nEnter your OpenAI API key: ").strip()
    elif provider_choice == "2":
        provider = "anthropic"
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            api_key = input("\nEnter your Anthropic API key: ").strip()
    else:
        print("❌ Invalid choice")
        sys.exit(1)
    
    if not api_key:
        print("❌ API key required")
        sys.exit(1)
    
    # Options
    print("\n" + "="*60)
    print("Enhancement Options:")
    print("="*60)
    add_tags = input("Extract tags for empty tag fields? (y/n): ").lower() == 'y'
    add_hints = input("Add semantic hints (is_definition, etc.)? (y/n): ").lower() == 'y'
    chunk_long = input("Chunk documents >1000 chars? (y/n): ").lower() == 'y'
    
    # Load documents
    doc_file = Path('data/splits/documents.json')
    if not doc_file.exists():
        print(f"❌ {doc_file} not found!")
        sys.exit(1)
    
    print(f"\n📖 Loading {doc_file}...")
    with open(doc_file, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    
    print(f"   Loaded {len(docs)} documents")
    
    # Filter documents that need enhancement
    docs_to_enhance = []
    for doc in docs:
        needs_tags = add_tags and not doc.get('tags')
        needs_chunking = chunk_long and len(doc.get('text', '')) > 1000
        needs_hints = add_hints and 'is_definition' not in doc
        
        if needs_tags or needs_chunking or needs_hints:
            docs_to_enhance.append(doc)
    
    if not docs_to_enhance:
        print("\n✅ All documents already enhanced!")
        return
    
    print(f"\n🔄 Enhancing {len(docs_to_enhance)} documents...")
    print(f"   (This may take a few minutes)")
    
    enhanced_docs = []
    docs_unchanged = [d for d in docs if d not in docs_to_enhance]
    
    for i, doc in enumerate(docs_to_enhance, 1):
        doc_id = doc.get('doc_id', 'unknown')
        print(f"\n[{i}/{len(docs_to_enhance)}] Processing {doc_id}...")
        
        try:
            result = enhance_document(doc, api_key, provider, add_tags, add_hints)
            enhanced_docs.extend(result)
            
            if len(result) > 1:
                print(f"  ✓ Chunked into {len(result)} parts")
            else:
                tags = result[0].get('tags', [])
                if tags:
                    print(f"  ✓ Tags: {tags}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            enhanced_docs.append(doc)  # Keep original
        
        # Rate limiting
        time.sleep(0.5)
    
    # Combine all documents
    all_docs = docs_unchanged + enhanced_docs
    
    print(f"\n📊 Results:")
    print(f"   Original: {len(docs)} documents")
    print(f"   Enhanced: {len(all_docs)} documents")
    print(f"   Added: {len(all_docs) - len(docs)} new chunks")
    
    # Save enhanced documents
    backup_file = doc_file.with_suffix('.json.backup')
    print(f"\n💾 Saving backup to {backup_file}...")
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saving enhanced documents to {doc_file}...")
    with open(doc_file, 'w', encoding='utf-8') as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)
    
    # Statistics
    print("\n✅ Enhancement complete!")
    print("\n📊 Final statistics:")
    tags_count = sum(1 for d in all_docs if d.get('tags'))
    hints_count = sum(1 for d in all_docs if 'is_definition' in d)
    chunks_count = sum(1 for d in all_docs if 'chunk_index' in d)
    
    print(f"   Documents with tags: {tags_count}/{len(all_docs)}")
    print(f"   Documents with semantic hints: {hints_count}/{len(all_docs)}")
    print(f"   Chunked documents: {chunks_count}/{len(all_docs)}")


if __name__ == '__main__':
    main()
