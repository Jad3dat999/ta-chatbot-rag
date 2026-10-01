#!/usr/bin/env python3
"""
Create train/val/test splits from new processed piazza_qa_pairs.json
"""

import json
from pathlib import Path
import random

random.seed(42)


def flatten_qa_pairs(qa_threads):
    """
    Flatten hierarchical Q&A structure into individual pairs
    
    Input format:
    {
      "question": "Main question",
      "answers": ["answer1", "answer2"],
      "followups": [
        {"question": "followup?", "answers": ["answer"]}
      ],
      "tags": ["tag1"],
      "source": "piazza"
    }
    
    Output format:
    {
      "question": "Question text",
      "answer": "Answer text",
      "source": "piazza",
      "tags": ["tag1"],
      "type": "main" or "followup"
    }
    """
    flat_pairs = []
    pair_id = 1
    
    for thread in qa_threads:
        main_question = thread['question']
        main_answers = thread.get('answers', [])
        followups = thread.get('followups', [])
        tags = thread.get('tags', [])
        source = thread.get('source', 'piazza')
        
        # Create pairs from main question
        if main_answers:
            for answer in main_answers:
                flat_pairs.append({
                    'question': main_question,
                    'answer': answer,
                    'source': source,
                    'tags': tags,
                    'type': 'main',
                    'pair_id': f'piazza_{pair_id}'
                })
                pair_id += 1
        else:
            # Include unanswered as well
            flat_pairs.append({
                'question': main_question,
                'answer': '',
                'source': source,
                'tags': tags + ['unanswered'],
                'type': 'main',
                'pair_id': f'piazza_{pair_id}'
            })
            pair_id += 1
        
        # Create pairs from followup questions
        for followup in followups:
            followup_question = followup['question']
            followup_answers = followup.get('answers', [])
            
            if followup_answers:
                for answer in followup_answers:
                    flat_pairs.append({
                        'question': followup_question,
                        'answer': answer,
                        'source': source,
                        'tags': tags,
                        'type': 'followup',
                        'parent_question': main_question,
                        'pair_id': f'piazza_{pair_id}'
                    })
                    pair_id += 1
    
    return flat_pairs


def load_synthetic_data():
    """Load synthetic assignment Q&A if exists"""
    synthetic_file = Path('data/processed_old/assignment_qa_pairs.json')
    
    if not synthetic_file.exists():
        print("  No synthetic assignment data found")
        return []
    
    with open(synthetic_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  Loaded {len(data)} synthetic assignment Q&A pairs")
    return data


def create_splits(flat_pairs, test_size=0.1, val_size=0.1):
    """
    Split data into train/val/test
    
    Args:
        flat_pairs: List of flat Q&A pairs
        test_size: Proportion for test set (default 10%)
        val_size: Proportion for validation set (default 10%)
    
    Returns:
        train, val, test splits
    """
    # Filter out unanswered questions for training
    answered_pairs = [p for p in flat_pairs if p['answer']]
    
    print(f"\nTotal Q&A pairs: {len(flat_pairs)}")
    print(f"Answered pairs: {len(answered_pairs)}")
    print(f"Unanswered: {len(flat_pairs) - len(answered_pairs)}")
    
    # Shuffle data
    random.shuffle(answered_pairs)
    
    # Calculate split indices
    n = len(answered_pairs)
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size / (1 - test_size)))
    
    # Split
    train = answered_pairs[:val_idx]
    val = answered_pairs[val_idx:test_idx]
    test = answered_pairs[test_idx:]
    
    return train, val, test


def main():
    print("="*60)
    print("Creating Train/Val/Test Splits from New Data")
    print("="*60)
    
    # Load new piazza data
    piazza_file = Path('data/processed/piazza_qa_pairs.json')
    
    if not piazza_file.exists():
        print(f"\n❌ Error: {piazza_file} not found!")
        print("   Please run process_raw_qa.py first")
        return
    
    print(f"\n📖 Loading data from {piazza_file}...")
    with open(piazza_file, 'r', encoding='utf-8') as f:
        qa_threads = json.load(f)
    
    print(f"   Loaded {len(qa_threads)} Q&A threads")
    
    # Flatten hierarchical structure
    print("\n🔄 Flattening hierarchical Q&A structure...")
    flat_pairs = flatten_qa_pairs(qa_threads)
    print(f"   Created {len(flat_pairs)} flat Q&A pairs")
    
    # Load synthetic assignment data if exists
    print("\n📖 Looking for synthetic assignment data...")
    synthetic_pairs = load_synthetic_data()
    
    # Combine all data
    all_pairs = flat_pairs + synthetic_pairs
    print(f"\n✅ Total dataset: {len(all_pairs)} Q&A pairs")
    
    # Create splits
    print("\n✂️ Creating splits...")
    train, val, test = create_splits(all_pairs, test_size=0.1, val_size=0.1)
    
    print(f"\n📊 Split sizes:")
    print(f"   Train: {len(train)} ({len(train)/len(all_pairs)*100:.1f}%)")
    print(f"   Val:   {len(val)} ({len(val)/len(all_pairs)*100:.1f}%)")
    print(f"   Test:  {len(test)} ({len(test)/len(all_pairs)*100:.1f}%)")
    
    # Save splits
    splits_dir = Path('data/splits')
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n💾 Saving splits...")
    
    with open(splits_dir / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved train.json")
    
    with open(splits_dir / 'val.json', 'w', encoding='utf-8') as f:
        json.dump(val, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved val.json")
    
    with open(splits_dir / 'test.json', 'w', encoding='utf-8') as f:
        json.dump(test, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved test.json")
    
    # Show sample
    print("\n" + "="*60)
    print("Sample Q&A Pair from Train Set")
    print("="*60)
    sample = train[0]
    print(f"Question: {sample['question'][:100]}...")
    print(f"Answer: {sample['answer'][:100]}...")
    print(f"Type: {sample['type']}")
    print(f"Tags: {sample['tags']}")
    
    # Statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    
    main_count = len([p for p in all_pairs if p.get('type') == 'main'])
    followup_count = len([p for p in all_pairs if p.get('type') == 'followup'])
    synthetic_count = len([p for p in all_pairs if p.get('type') == 'synthetic'])
    
    print(f"Main questions: {main_count}")
    print(f"Followup questions: {followup_count}")
    
    if synthetic_count > 0:
        print(f"Synthetic pairs: {synthetic_count}")
    
    print("\n" + "="*60)
    print("✅ Splits created successfully!")
    print("="*60)
    print("\nReady for training! 🚀")


if __name__ == '__main__':
    main()
