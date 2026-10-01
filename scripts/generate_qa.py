import json
import re
import csv
from pathlib import Path

# Common words/phrases to filter out
STOP_PHRASES = {
    'about me', 'rice canvas', 'signing special', 'special registration',
    'email me', 'cc marci', 'marci wilson', 'rice netid', 'netid',
    'canvas', 'piazza', 'assignment', 'lecture', 'slide', 'page',
    'due date', 'deadline', 'submission', 'instructions', 'requirements',
    'ballistic missile', 'missile defense', 'frequency trading', 'high frequency',
    'lincoln laboratory', 'baylor college', 'rice university', 'harvard',
    'applied mathematics', 'computer science', 'education ph', 'ph.d',
    'continuum analytics', 'new faculty', 'rich baraniuk', 'ankit lab'
}

# Common stop words (single words that aren't meaningful concepts)
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
    'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
    'every', 'some', 'any', 'no', 'not', 'more', 'most', 'many', 'much',
    'few', 'little', 'other', 'another', 'such', 'only', 'just', 'also',
    'very', 'too', 'so', 'than', 'then', 'there', 'here', 'where', 'now'
}

def clean_text(text):
    """Clean text while preserving structure"""
    if not text:
        return ""
    # Preserve newlines for code blocks and formatting
    text = re.sub(r'[ \t]+', ' ', text)  # Only collapse spaces/tabs, not newlines
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    return text.strip()

def is_meaningful_phrase(phrase):
    """Check if a phrase is meaningful (not a stop phrase or common word)"""
    phrase_lower = phrase.lower()
    
    # Filter out stop phrases
    for stop_phrase in STOP_PHRASES:
        if stop_phrase in phrase_lower:
            return False
    
    # Filter out if all words are stop words
    words = phrase.split()
    if all(word.lower() in STOP_WORDS for word in words):
        return False
    
    # Filter out very short phrases (less than 4 chars)
    if len(phrase.replace(' ', '')) < 4:
        return False
    
    # Filter out phrases that are just numbers or single letters
    if re.match(r'^[A-Z]$', phrase) or re.match(r'^\d+$', phrase):
        return False
    
    # Filter out email-like patterns
    if '@' in phrase or '.com' in phrase_lower or '.edu' in phrase_lower:
        return False
    
    # Filter out date-like patterns
    if re.search(r'\d{1,2}[/-]\d{1,2}', phrase):
        return False
    
    return True

def extract_meaningful_concepts(text):
    """Extract meaningful technical concepts and terms"""
    concepts = set()
    
    # Common technical terms in deep learning (prioritize these)
    technical_terms = [
        'Neural Network', 'Neural Networks', 'Deep Learning', 'Machine Learning',
        'Convolutional Neural Network', 'CNN', 'Recurrent Neural Network', 'RNN',
        'Long Short-Term Memory', 'LSTM', 'Gated Recurrent Unit', 'GRU',
        'Backpropagation', 'Gradient Descent', 'Stochastic Gradient Descent', 'SGD',
        'Adam Optimizer', 'ReLU', 'Activation Function', 'Loss Function',
        'Overfitting', 'Regularization', 'Dropout', 'Batch Normalization',
        'Transfer Learning', 'Fine-tuning', 'Feature Extraction', 'Embedding',
        'Attention Mechanism', 'Transformer', 'Self-Attention', 'Multi-Head Attention',
        'Generative Adversarial Network', 'GAN', 'Variational Autoencoder', 'VAE',
        'Autoencoder', 'Encoder', 'Decoder', 'Latent Space', 'Representation Learning'
    ]
    
    # Check for known technical terms first
    text_lower = text.lower()
    for term in technical_terms:
        if term.lower() in text_lower:
            concepts.add(term)
    
    # Pattern 1: Technical terms (Capitalized multi-word phrases, but stop at sentence boundaries)
    # Look for patterns like "Neural Network" but avoid "Neural Networks Takes"
    # Use word boundaries and avoid capturing verbs after the phrase
    pattern1 = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})(?:\s+(?:Takes|Returns|Is|Are|Was|Were|Has|Have|Do|Does|Did|Will|Can|Should|May|Might))?\b'
    matches1 = re.findall(pattern1, text)
    for match in matches1:
        # Skip if it's a known stop phrase or if it ends with a verb-like word
        if is_meaningful_phrase(match) and not match.endswith(('Takes', 'Returns', 'Is', 'Are')):
            # Prefer shorter, more specific phrases
            if len(match.split()) <= 3:
                concepts.add(match)
    
    # Pattern 2: Acronyms and technical abbreviations
    # Examples: "CNN", "RNN", "LSTM", "GPU", "API"
    pattern2 = r'\b([A-Z]{2,6})\b'
    matches2 = re.findall(pattern2, text)
    for match in matches2:
        # Filter out common non-technical acronyms
        if match not in ['PDF', 'URL', 'HTTP', 'HTTPS', 'HTML', 'CSS', 'JS', 'ID', 'CC', 'MIT', 'PhD']:
            concepts.add(match)
    
    # Pattern 3: Technical terms with numbers (but be more selective)
    # Examples: "Layer 3", "Python 3", but avoid "Version 2.0" unless it's meaningful
    pattern3 = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+\d+)\b'
    matches3 = re.findall(pattern3, text)
    for match in matches3:
        # Only include if it's a meaningful technical term
        base_term = re.sub(r'\s+\d+$', '', match)
        if is_meaningful_phrase(base_term) and len(base_term.split()) <= 2:
            concepts.add(match)
    
    # Filter and return top concepts
    filtered_concepts = [c for c in concepts if is_meaningful_phrase(c)]
    # Prioritize known technical terms
    known_terms = [c for c in filtered_concepts if c in technical_terms]
    other_terms = [c for c in filtered_concepts if c not in technical_terms]
    # Sort by length (longer phrases are usually more specific)
    known_terms.sort(key=len, reverse=True)
    other_terms.sort(key=len, reverse=True)
    # Return up to 3 known terms + 2 other terms = max 5 concepts
    return (known_terms[:3] + other_terms[:2])[:5]

def generate_question_variations(base_question, concept=None):
    """Generate variations of questions for diversity"""
    variations = [base_question]
    
    if concept:
        variations.extend([
            f"Can you explain {concept.lower()} from the lecture?",
            f"What is {concept.lower()}?",
            f"Tell me about {concept.lower()}.",
        ])
    
    return variations

def generate_csv():
    """Generate synthetic Q&A pairs from slides and assignments"""
    # Load your data files
    try:
        with open('data/processed/piazza_qa.json', 'r', encoding='utf-8') as f:
            piazza_data = json.load(f)
        with open('data/processed/all_data.json', 'r', encoding='utf-8') as f:
            content_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return

    generated_qa = []
    stats = {
        'lecture_questions': 0,
        'assignment_questions': 0,
        'concept_questions': 0
    }

    # --- Process Lectures ---
    if 'lectures' in content_data:
        for lecture in content_data['lectures']:
            lecture_title = lecture.get('title', 'Unknown Lecture')
            lecture_num = lecture.get('lecture_number', '')
            
            for slide in lecture.get('slides', []):
                page_num = slide.get('page', '?')
                raw_text = slide.get('text', '')
                text = clean_text(raw_text)
                
                # Skip slides with too little content
                if len(text) < 30: 
                    continue
                
                # Skip slides that are mostly figures/equations with minimal text
                content_type = slide.get('content_type', 'text')
                if content_type in ['figure', 'equation', 'minimal'] and len(text) < 50:
                    continue
                
                # Template 1: Direct question about slide content
                q1 = f"What is covered on slide {page_num} of '{lecture_title}'?"
                a1 = f"Slide {page_num} of {lecture_title} covers:\n\n{text}"
                generated_qa.append([q1, a1, f"{lecture_title} - Slide {page_num}"])
                stats['lecture_questions'] += 1
                
                # Template 2: Help request (only for substantial slides)
                if len(text) > 100:
                    q2 = f"Can you help me understand slide {page_num} from {lecture_title}?"
                    a2 = f"Certainly! Slide {page_num} of {lecture_title} explains:\n\n{text}"
                    generated_qa.append([q2, a2, f"{lecture_title} - Slide {page_num}"])
                    stats['lecture_questions'] += 1
                
                # Template 3: Concept-based questions (improved)
                concepts = extract_meaningful_concepts(text)
                for concept in concepts:
                    # Only create concept questions if the concept appears meaningfully in the text
                    if concept.lower() in text.lower():
                        q3 = f"What does {lecture_title} say about {concept}?"
                        # Extract context around the concept
                        concept_lower = concept.lower()
                        text_lower = text.lower()
                        idx = text_lower.find(concept_lower)
                        if idx != -1:
                            # Get surrounding context (about 200 chars)
                            start = max(0, idx - 100)
                            end = min(len(text), idx + len(concept) + 100)
                            context = text[start:end]
                            a3 = f"In {lecture_title}, {concept} is discussed on slide {page_num}. Here's the relevant context:\n\n{context}"
                        else:
                            a3 = f"On slide {page_num} of {lecture_title}, {concept} is mentioned in this context:\n\n{text}"
                        
                        generated_qa.append([q3, a3, f"{lecture_title} - Slide {page_num} - {concept}"])
                        stats['concept_questions'] += 1

    # --- Process Assignments ---
    if 'assignments' in content_data:
        for assignment in content_data['assignments']:
            assign_title = assignment.get('title', 'Unknown Assignment')
            assign_num = assignment.get('assignment_number', '')
            
            for section in assignment.get('sections', []):
                sec_title = section.get('title', 'General')
                content = clean_text(section.get('content', ''))
                
                # Skip very short sections
                if len(content) < 20: 
                    continue
                
                # Skip introduction sections that are too generic
                if sec_title.lower() == 'introduction' and len(content) < 50:
                    continue
                
                # Template 4: Requirements question
                q4 = f"What are the requirements for '{sec_title}' in {assign_title}?"
                a4 = f"For the section '{sec_title}' in {assign_title}, the requirements are:\n\n{content}"
                generated_qa.append([q4, a4, f"{assign_title} - {sec_title}"])
                stats['assignment_questions'] += 1
                
                # Template 5: Student guidance (only for substantial sections)
                if len(content) > 50:
                    q5 = f"I need help with '{sec_title}' in {assign_title}. What should I know?"
                    a5 = f"For '{sec_title}' in {assign_title}, here's what you need to know:\n\n{content}"
                    generated_qa.append([q5, a5, f"{assign_title} - {sec_title}"])
                    stats['assignment_questions'] += 1
                
                # Template 6: Deadline extraction (only if dates found)
                dates = re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,?\s+\d{4})?', content, re.IGNORECASE)
                if dates:
                    for date in dates[:1]:  # Limit to one date question per section
                        q6 = f"When is the deadline for {assign_title} section '{sec_title}'?"
                        a6 = f"The deadline mentioned for '{sec_title}' in {assign_title} is: {date}. Full details:\n\n{content}"
                        generated_qa.append([q6, a6, f"{assign_title} - {sec_title} - Deadline"])
                        stats['assignment_questions'] += 1
                
                # Template 7: Task-specific questions (for sections that are tasks)
                if sec_title.startswith('Task') or 'task' in sec_title.lower():
                    q7 = f"What do I need to do for {sec_title} in {assign_title}?"
                    a7 = f"For {sec_title} in {assign_title}, you need to:\n\n{content}"
                    generated_qa.append([q7, a7, f"{assign_title} - {sec_title}"])
                    stats['assignment_questions'] += 1

    # --- Write to CSV ---
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / 'synthetic_qa_dataset.csv'
    
    with open(output_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Question', 'Answer', 'Source'])
        writer.writerows(generated_qa)

    print(f"\n{'='*60}")
    print(f"✅ Success! Generated {len(generated_qa)} Q&A pairs")
    print(f"{'='*60}")
    print(f"\n📊 Statistics:")
    print(f"   Lecture questions: {stats['lecture_questions']}")
    print(f"   Concept questions: {stats['concept_questions']}")
    print(f"   Assignment questions: {stats['assignment_questions']}")
    print(f"   Total: {len(generated_qa)}")
    print(f"\n💾 Saved to: {output_filename}")

if __name__ == "__main__":
    generate_csv()
