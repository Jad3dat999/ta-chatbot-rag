#!/usr/bin/env python3
"""
Process raw_qa.txt into clean JSON format
Extracts Q&A pairs from Piazza API responses
"""

import json
import re
import html
from pathlib import Path
from html.parser import HTMLParser


class HTMLStripper(HTMLParser):
    """Strip HTML tags and decode entities"""
    def __init__(self):
        super().__init__()
        self.text = []
    
    def handle_data(self, data):
        self.text.append(data)
    
    def get_text(self):
        return ' '.join(self.text).strip()


def strip_html(html_content):
    """Convert HTML to plain text"""
    if not html_content:
        return ''
    
    # Decode HTML entities first
    text = html.unescape(html_content)
    
    # Remove HTML tags
    stripper = HTMLStripper()
    stripper.feed(text)
    clean_text = stripper.get_text()
    
    # Clean up extra whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = clean_text.strip()
    
    return clean_text


def parse_json_objects(file_path):
    """Parse multiple JSON objects from a file (separated by blank lines)"""
    json_objects = []
    
    # Read entire file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by pattern: }\n\n{ (end of one JSON, blank line, start of next)
    # Use lookahead/lookbehind to keep the braces
    parts = re.split(r'\}\s*\n\s*\n\s*\{', content)
    
    # Parse each JSON object
    for i, part in enumerate(parts):
        # First part: add closing brace
        if i == 0:
            json_str = part.rstrip() + '\n}'
        # Last part: add opening brace
        elif i == len(parts) - 1:
            json_str = '{\n' + part.lstrip()
        # Middle parts: add both braces
        else:
            json_str = '{\n' + part + '\n}'
        
        try:
            obj = json.loads(json_str)
            json_objects.append(obj)
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error at object {i+1}: {e}")
            # Try to find the issue
            lines = json_str.split('\n')
            if len(lines) > 3:
                print(f"   Lines 1-3: {lines[:3]}")
                print(f"   Lines -3: {lines[-3:]}")
            # Try alternative: maybe it's already complete JSON
            try:
                obj = json.loads(part)
                json_objects.append(obj)
                print(f"   ✓ Fixed by parsing without braces")
            except:
                pass
    
    return json_objects


def extract_children_recursive(children_list):
    """Recursively extract followup Q&A from children"""
    followups = []
    
    if not children_list:
        return followups
    
    for child in children_list:
        child_type = child.get('type', '')
        subject_raw = child.get('subject', '').strip()
        
        if not subject_raw:
            continue
        
        # Strip HTML from subject
        subject = strip_html(subject_raw)
        
        # Followup = question, Feedback = answer
        if child_type == 'followup':
            # This is a followup question
            # Get answers from its children (feedbacks)
            answers = []
            child_children = child.get('children', [])
            
            for feedback in child_children:
                if feedback.get('type') == 'feedback':
                    feedback_subject_raw = feedback.get('subject', '').strip()
                    if feedback_subject_raw:
                        feedback_subject = strip_html(feedback_subject_raw)
                        if feedback_subject:
                            answers.append(feedback_subject)
            
            # Create followup Q&A entry
            followup_entry = {
                'question': subject,
                'answer': ' '.join(answers) if answers else '(No answer yet)',
                'created': child.get('created', ''),
                'followup_id': child.get('id', ''),
                'num_answers': len(answers)
            }
            
            # Recursively extract nested children
            nested_followups = extract_children_recursive(child_children)
            if nested_followups:
                followup_entry['followups'] = nested_followups
            
            followups.append(followup_entry)
        
        elif child_type == 'feedback':
            # This is a direct feedback/answer (might be nested)
            # We'll handle these as part of followup processing
            pass
    
    return followups


def extract_qa_from_response(response):
    """Extract Q&A from a Piazza API response, including followups"""
    if not response.get('result'):
        return None
    
    result = response['result']
    
    # Get latest version from history
    history = result.get('history', [])
    if not history:
        return None
    
    latest = history[0]
    
    # Extract main question and answer
    question = latest.get('subject', '').strip()
    answer_html = latest.get('content', '')
    answer = strip_html(answer_html)
    
    # Skip if no question or answer too short
    if not question or len(answer) < 20:
        return None
    
    # Extract followup discussions
    children = result.get('children', [])
    followups = extract_children_recursive(children)
    
    # Build Q&A entry
    qa_entry = {
        'question': question,
        'answer': answer,
        'source': f"piazza.com/class/<class_id>/post/{result.get('nr', 'unknown')}",
        'post_id': result.get('id', ''),
        'post_number': result.get('nr', ''),
        'tags': result.get('folders', []),
        'type': result.get('type', 'unknown'),
        'views': result.get('unique_views', 0),
        'created': result.get('created', ''),
        'history_size': result.get('history_size', 0),
        'num_followups': len(followups)
    }
    
    # Add followups if any
    if followups:
        qa_entry['followups'] = followups
    
    return qa_entry


def main():
    print("🚀 Processing raw_qa.txt...\n")
    
    # Input and output paths
    input_file = Path('data/raw/raw_qa.txt')
    output_file = Path('data/raw/piazza_qa.json')
    
    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        return
    
    print(f"📖 Reading {input_file}...")
    
    # Parse all JSON objects
    json_objects = parse_json_objects(input_file)
    print(f"✓ Parsed {len(json_objects)} JSON objects\n")
    
    # Extract Q&A pairs
    print("📝 Extracting Q&A pairs...")
    qa_pairs = []
    
    for i, obj in enumerate(json_objects, 1):
        qa = extract_qa_from_response(obj)
        if qa:
            qa_pairs.append(qa)
            if i % 10 == 0:
                print(f"  Processed {i}/{len(json_objects)}... ({len(qa_pairs)} valid)")
    
    print(f"\n✓ Extracted {len(qa_pairs)} Q&A pairs\n")
    
    # Save to JSON file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved to {output_file}\n")
    
    # Statistics
    print("📊 Statistics:")
    print(f"   Total Q&A pairs: {len(qa_pairs)}")
    if qa_pairs:
        avg_q_len = sum(len(p['question']) for p in qa_pairs) / len(qa_pairs)
        avg_a_len = sum(len(p['answer']) for p in qa_pairs) / len(qa_pairs)
        print(f"   Avg question length: {int(avg_q_len)} chars")
        print(f"   Avg answer length: {int(avg_a_len)} chars")
        print(f"   Total views: {sum(p['views'] for p in qa_pairs)}")
        
        # Count followups
        total_followups = sum(p.get('num_followups', 0) for p in qa_pairs)
        posts_with_followups = sum(1 for p in qa_pairs if p.get('num_followups', 0) > 0)
        print(f"   Posts with followups: {posts_with_followups}/{len(qa_pairs)}")
        print(f"   Total followup discussions: {total_followups}")
        
        # Post types
        types = {}
        for p in qa_pairs:
            types[p['type']] = types.get(p['type'], 0) + 1
        print(f"   Post types: {types}")
        
        # Sample entries
        print(f"\n📝 Sample entries:")
        for i, entry in enumerate(qa_pairs[:3], 1):
            print(f"\n   {i}. Post #{entry['post_number']}")
            print(f"      Q: {entry['question'][:70]}...")
            print(f"      A: {entry['answer'][:70]}...")
            if entry.get('num_followups', 0) > 0:
                print(f"      Followups: {entry['num_followups']}")
                # Show first followup
                first_followup = entry['followups'][0]
                print(f"        - Q: {first_followup['question'][:60]}...")
                print(f"          A: {first_followup['answer'][:60]}...")
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()

