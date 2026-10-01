"""
Verify data format and quality

Usage:
    python scripts/verify_data.py --input data/raw/piazza_qa.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict


def verify_piazza_data(file_path: Path) -> bool:
    """Verify Piazza Q&A data format"""
    
    print(f"Verifying {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    if not isinstance(data, list):
        print("❌ Data should be a list of Q&A pairs")
        return False
    
    if len(data) == 0:
        print("❌ No data found")
        return False
    
    print(f"✓ Found {len(data)} Q&A pairs")
    
    # Check each entry
    required_fields = ['question', 'answer']
    optional_fields = ['source', 'tags']
    
    errors = []
    warnings = []
    
    for i, entry in enumerate(data):
        # Check required fields
        for field in required_fields:
            if field not in entry:
                errors.append(f"Entry {i}: Missing required field '{field}'")
            elif not entry[field] or not entry[field].strip():
                errors.append(f"Entry {i}: Empty '{field}' field")
        
        # Check optional fields
        if 'source' not in entry:
            warnings.append(f"Entry {i}: Missing 'source' field")
        
        if 'tags' not in entry:
            warnings.append(f"Entry {i}: Missing 'tags' field")
        elif not isinstance(entry['tags'], list):
            errors.append(f"Entry {i}: 'tags' should be a list")
    
    # Print results
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        return False
    
    if warnings:
        print(f"\n⚠️  Found {len(warnings)} warnings:")
        for warning in warnings[:10]:
            print(f"  - {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")
    
    # Statistics
    print("\n📊 Statistics:")
    print(f"  Total Q&A pairs: {len(data)}")
    
    avg_q_len = sum(len(e['question']) for e in data) / len(data)
    avg_a_len = sum(len(e['answer']) for e in data) / len(data)
    print(f"  Avg question length: {avg_q_len:.0f} characters")
    print(f"  Avg answer length: {avg_a_len:.0f} characters")
    
    # Count tags
    all_tags = []
    for entry in data:
        if 'tags' in entry and isinstance(entry['tags'], list):
            all_tags.extend(entry['tags'])
    
    if all_tags:
        unique_tags = set(all_tags)
        print(f"  Unique tags: {len(unique_tags)}")
        from collections import Counter
        top_tags = Counter(all_tags).most_common(5)
        print(f"  Top tags: {', '.join(f'{tag}({count})' for tag, count in top_tags)}")
    
    # Show examples
    print("\n📝 Sample entries:")
    for i in range(min(2, len(data))):
        print(f"\n  Entry {i+1}:")
        print(f"    Q: {data[i]['question'][:100]}...")
        print(f"    A: {data[i]['answer'][:100]}...")
        if 'tags' in data[i]:
            print(f"    Tags: {data[i]['tags']}")
    
    print("\n✓ Data format is valid!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify data format")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file to verify"
    )
    
    args = parser.parse_args()
    
    file_path = Path(args.input)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        exit(1)
    
    success = verify_piazza_data(file_path)
    exit(0 if success else 1)

