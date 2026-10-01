#!/usr/bin/env python3
"""
Piazza Q&A Extractor using Python
Uses cookies from your browser to access Piazza API

SETUP:
1. Install: pip install requests browser-cookie3
2. Run: python piazza_python_extractor.py
"""

import json
import os
import time
import re
from pathlib import Path

try:
    import requests
    import browser_cookie3
except ImportError:
    print("❌ Missing dependencies!")
    print("Install: pip install requests browser-cookie3")
    exit(1)

CLASS_ID = os.environ["PIAZZA_CLASS_ID"]  # your course's Piazza network id
API_URL = 'https://piazza.com/logic/api'

# Post IDs and numbers from the feed
# Format: (mongodb_id, post_number)
POSTS = [
    ('mivgar0as4y3t6', '141'),
    ('mhukhfnhwmj3ot', '111'),
    ('mhsa12hzjtz7ht', '109'),
    ('mglecbwjzza2ui', '33'),
    ('mgk2nojvzhi5yl', '32'),
    ('mgbh8owxo6v1lu', '25'),
    ('mf41eodoy0s46w', '5'),
    ('miw97mfder94q8', '144'),
    ('miw92vw3xoh3gc', '143'),
    ('mins4rjl74d2fc', '135'),
    ('minieax6ypp5ss', '133'),
    ('minht2j2eyp44z', '132'),
    ('mi9gv82ffwp17j', '127'),
    ('mi4x1f3tx6j1m2', '122'),
    ('mi3r62m0ax431m', '121'),
    ('mi3pvlzpsbo65o', '120'),
    ('mi3ly347rgf37z', '119'),
    ('mi0pwy6dw9l2gx', '115'),
    ('mhuwf9ob4jy3zy', '114'),
    ('mhuw9z9ffn34qd', '113'),
    ('mhuugd7ev0d2v1', '112'),
    ('mhprodqh6pv5co', '108'),
    ('mhpmhfv71w846o', '107'),
    ('mhod9r4k7ow2j4', '105'),
    ('mhnv90pm5anzg', '104'),
    ('mhns9kt3wqr5tt', '102'),
    ('mhn7oseeo8i4bf', '101'),
    ('mhn5tgrowxh7p3', '100'),
    ('mhn3i8922cq2qu', '99'),
    ('mhlnztcb19k75w', '92'),
    ('mhlghhlz8h348t', '91'),
    ('mhjzky1devc49t', '89'),
    ('mhjysovq6cz2eh', '88'),
    ('mhjpk4437g2642', '87'),
    ('mhjd81huzri5tz', '85'),
    ('mhinsw1vbgv29r', '84'),
    ('mhfmq7zigwn6xu', '82'),
    ('mhdzmd5rt7237p', '79'),
    ('mh8s10sg8jf1a0', '75'),
    ('mh8emd6d6e2ku', '74'),
    ('mh55nybpeto1ch', '66'),
    ('mh45o5km4470r', '65'),
    ('mh3xemjdi9w2ax', '64'),
    ('mgtx2tx5u84343', '62'),
    ('mgsmunfefnaij', '61'),
    ('mgshlxx4qe66kt', '60'),
    ('mgsb9dk0rtb3cg', '58'),
    ('mgs97t399552kq', '57'),
    ('mgrkpc49u8eqk', '56'),
    ('mgr6zec9yn21cj', '55'),
    ('mgr20n71xuy1d8', '52'),
    ('mgqznli7us645d', '51'),
    ('mgqv61wochskx', '49'),
    ('mgqulxoo4142kg', '48'),
    ('mgqtu9eyuss1bv', '47'),
    ('mgqssm3ew5i5rv', '46'),
    ('mgpudi2q88b180', '44'),
    ('mgplar8kk7m2jo', '43'),
    ('mgpe0mreql62vy', '42'),
    ('mgp1g6h1frj2pz', '41'),
    ('mgogml9zkax57o', '40'),
    ('mgo8kl827n61hl', '39'),
    ('mgnunfavqz63z', '37'),
    ('mgnac4cyys26ml', '36'),
    ('mgn9vlew3nm5gg', '35'),
    ('mgmrdg19bld47r', '34'),
    ('mgicdc5v3106s4', '31'),
    ('mgiaz7e4jpf1v6', '30'),
    ('mghawwbouza6cd', '28'),
    ('mggruruy7e630i', '26'),
    ('mg8t890ifb51sa', '24'),
    ('mg8dgge0hv12xi', '23'),
    ('mg8ait8co4g1kv', '22'),
    ('mg7fsb3zubxct', '21'),
    ('mg59b0cplsb647', '20'),
    ('mg0bqkmgcuc3m1', '19'),
    ('mfzksf0u4u8tk', '18'),
    ('mfypbohwhtbuv', '17'),
    ('mfllbozssndqa', '14'),
    ('mfjs2keh8r6j7', '9'),
    ('mfj2zgr42cs7fh', '8'),
    ('mf41eo21r4o45o', '1'),
]


def strip_html(html_text):
    """Remove HTML tags from text"""
    clean = re.sub('<.*?>', '', html_text)
    return clean.strip()


def get_browser_cookies():
    """Try to load cookies from browser"""
    print("🔍 Loading cookies from browser...")
    
    # Try Chrome first, then other browsers
    try:
        cookies = browser_cookie3.chrome(domain_name='piazza.com')
        cookie_list = list(cookies)
        if cookie_list:
            print(f"✓ Loaded {len(cookie_list)} Chrome cookies")
            return cookies
        else:
            print("⚠️  Chrome cookies empty")
    except Exception as e:
        print(f"⚠️  Chrome: {e}")
    
    try:
        cookies = browser_cookie3.firefox(domain_name='piazza.com')
        cookie_list = list(cookies)
        if cookie_list:
            print(f"✓ Loaded {len(cookie_list)} Firefox cookies")
            return cookies
    except Exception as e:
        print(f"⚠️  Firefox: {e}")
    
    try:
        cookies = browser_cookie3.safari(domain_name='piazza.com')
        cookie_list = list(cookies)
        if cookie_list:
            print(f"✓ Loaded {len(cookie_list)} Safari cookies")
            return cookies
    except Exception as e:
        print(f"⚠️  Safari: {e}")
    
    print("\n❌ Could not load Piazza cookies from any browser")
    print("Make sure you're logged in to Piazza in your browser")
    return None


def fetch_post(session, post_id, post_number):
    """Fetch a single post using Piazza API"""
    
    payload = {
        'method': 'content.get',
        'params': {
            'cid': post_id,
            'nid': CLASS_ID,
            'student_view': None
        }
    }
    
    try:
        response = session.post(API_URL, json=payload, timeout=10)
        data = response.json()
        
        if data.get('error'):
            return None, data['error']
        
        result = data.get('result')
        if not result or not result.get('history'):
            return None, "No content"
        
        # Get latest version
        latest = result['history'][0]
        question = latest.get('subject', '')
        answer = strip_html(latest.get('content', ''))
        
        if not question or not answer or len(answer) < 20:
            return None, "Too short"
        
        post_data = {
            'question': question,
            'answer': answer,
            'source': f'piazza.com/class/{CLASS_ID}/post/{post_number}',
            'post_id': post_id,
            'post_number': post_number,
            'tags': result.get('folders', []),
            'type': result.get('type', 'unknown'),
            'views': result.get('unique_views', 0),
            'created': result.get('created', '')
        }
        
        return post_data, None
        
    except Exception as e:
        return None, str(e)


def main():
    print("🚀 Piazza Python Extractor\n")
    
    # Load browser cookies
    cookies = get_browser_cookies()
    if cookies is None:
        print("\n💡 Alternative: Ask your instructor for Piazza data export")
        return
    
    # Create session with cookies
    session = requests.Session()
    session.cookies = cookies
    session.headers.update({
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })
    
    print(f"\n📝 Fetching {len(POSTS)} posts...\n")
    
    all_data = []
    success_count = 0
    
    for i, (post_id, post_num) in enumerate(POSTS, 1):
        print(f"[{i}/{len(POSTS)}] Post {post_num} ({post_id})...", end=' ')
        
        post_data, error = fetch_post(session, post_id, post_num)
        
        if post_data:
            all_data.append(post_data)
            success_count += 1
            print(f"✓ {post_data['question'][:50]}...")
        else:
            print(f"⚠️  {error}")
        
        # Rate limiting
        time.sleep(0.3)
        
        if i % 10 == 0:
            print(f"\n--- {success_count} successful ---\n")
    
    print("\n" + "="*60)
    print(f"✓ DONE! Extracted {success_count}/{len(POSTS)} posts")
    print("="*60)
    
    if all_data:
        # Save to file
        output_file = Path('data/raw/piazza_qa_python.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved to {output_file}")
        print(f"\n📊 Summary:")
        print(f"   Posts: {len(all_data)}")
        avg_len = sum(len(p['answer']) for p in all_data) / len(all_data)
        print(f"   Avg answer length: {int(avg_len)} chars")
        print(f"   Total views: {sum(p['views'] for p in all_data)}")
    else:
        print("\n❌ No posts extracted")
        print("\n💡 Suggestions:")
        print("1. Make sure you're logged in to Piazza in your browser")
        print("2. Try closing your browser and logging in again")
        print("3. Ask your instructor for official Piazza data export")


if __name__ == '__main__':
    main()

