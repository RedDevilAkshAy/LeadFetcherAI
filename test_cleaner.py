import re

def clean_owner_name(raw_text: str, business_name: str = '') -> str:
    if not raw_text:
        return 'Unknown'
    text = raw_text.strip().strip('*_`"\'')
    if not text or 'unknown' in text.lower():
        return 'Unknown'

    # Remove markdown citations like [1], [2]
    text = re.sub(r'\[\d+\]', '', text).strip()

    # Prefix cleanup (e.g. "Owner: John", "Founder: Jane")
    text = re.sub(r'^(?:owner|founder|ceo|director|in charge|name)\s*:\s*', '', text, flags=re.I).strip()

    # Pattern 1: '<Business> was founded/co-founded/created/established/owned by <Name>...'
    # or 'Founded by <Name>...'
    m = re.search(r'(?:(?:was|is)\s+)?(?:co-)?(?:founded|owned|created|established|started|operated|run|led)\s+by\s+([A-Z][a-zA-Z\.\s,\'-]+)', text, flags=re.I)
    if m:
        candidate = m.group(1).strip()
        candidate = re.split(r'\b(?:in\s+\d{4}|who|and\s+serves|alongside|with|as\s+well|from)\b|[;\.\n]', candidate, flags=re.I)[0].strip()
        candidate = re.sub(r'[,\.\s]+$', '', candidate).strip()
        if candidate and len(candidate.split()) <= 6:
            return candidate[:50]

    # Pattern 2: 'The (owner|founder|ceo|director) of <Business> is <Name>...'
    # or 'The (owner|founder) is <Name>...'
    m = re.search(r'(?:the\s+)?(?:owner|founder|ceo|director|person in charge|medical director)(?:\s+(?:and|&)\s+(?:founder|owner|ceo|director))?(?:\s+of\s+[^,]+)?\s+(?:is|was)\s+([A-Z][a-zA-Z\.\s,\'-]+)', text, flags=re.I)
    if m:
        candidate = m.group(1).strip()
        candidate = re.split(r'\b(?:who|and\s+serves|in\s+\d{4})\b|[;\.\n]', candidate, flags=re.I)[0].strip()
        candidate = re.sub(r'[,\.\s]+$', '', candidate).strip()
        if candidate and len(candidate.split()) <= 6:
            return candidate[:50]

    # Pattern 3: '<Name> is the (founder|owner|ceo|director|creator)...'
    # or '<Name> is a founder...'
    # or '<Name> serves as the...'
    # or '<Name> (founded|co-founded|owns|operates)...'
    m = re.match(r'^([A-Z][a-zA-Z\.\s,\'-]+?)\s+(?:(?:is|serves as|acts as)\s+(?:the\s+|an?\s+)?(?:co-)?(?:founder|owner|ceo|director|president|creator|medical director|proprietor)|(?:founded|co-founded|owns|operates|runs|started))\b', text, flags=re.I)
    if m:
        candidate = m.group(1).strip()
        candidate = re.sub(r'[,\.\s]+$', '', candidate).strip()
        if candidate and len(candidate.split()) <= 6:
            return candidate[:50]

    # Pattern 4: '<Name>, (the )?(founder|owner|ceo)...'
    m = re.match(r'^([A-Z][a-zA-Z\.\s,\'-]+?),\s*(?:the\s+|an?\s+)?(?:co-)?(?:founder|owner|ceo|director|president|medical director)\b', text, flags=re.I)
    if m:
        candidate = m.group(1).strip()
        candidate = re.sub(r'[,\.\s]+$', '', candidate).strip()
        if candidate and len(candidate.split()) <= 6:
            return candidate[:50]

    # If text is already a clean name (e.g. 1 to 5 words, no sentence verbs like 'is', 'was', 'the')
    cleaned = re.sub(r'(?<!\bDr)(?<!\bMr)(?<!\bMs)\.+$', '', text).strip()
    words = cleaned.split()
    if len(words) <= 5 and not any(w.lower() in {'is', 'was', 'are', 'were', 'the', 'of', 'in', 'at', 'for', 'founded', 'owns'} for w in words):
        return cleaned[:50].strip()

    # Fallback: take first clause before punctuation
    first_clause = re.split(r'[\.\n;]', text)[0].strip()
    if len(first_clause.split()) <= 5:
        return first_clause[:50].strip()

    return 'Unknown'


test_cases = [
    'Julia Pernas is the founder of Brickell Cosmetic C',
    'Lior Masaphor is the founder of No Filter Medical',
    'VIIV WELLNESS HAUS was founded by Lunmey in 2020',
    'VIIV WELLNESS HAUS was founded by Lunmey',
    'Paul Charette is the founder and CEO of Charett...',
    'Ava Franzoni',
    'Odette Binker',
    'Dr. Tali Arviv',
    'Jose Almeida',
    'Dr. Sabrina Gonzalez, DNP, APRN',
    'Justin Jeter, MD',
    'The owner of Miami Lakes Med Spa is Dr. Sabrina Gonzalez',
    'Founded by Nicole Habib.',
    'Unknown',
    'The founder is Dr. Alex Turner, who opened it in 2015.',
    'Michael Scott, founder and CEO of Dunder Mifflin',
]

for t in test_cases:
    print(f'{t!r:60} -> {clean_owner_name(t)!r}')
