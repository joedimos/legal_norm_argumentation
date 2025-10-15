from typing import List, Dict, Set, Optional
import re
from models import nlp, load_transformer_classifier

import pyreason as pr


def extract_norms(text: str, use_transformer: bool = False) -> List[Dict]:
    """
    Extract legal norms from text.
    
    Parameters
    ----------
    text : str
        Legal text to analyze
    use_transformer : bool
        Whether to use transformer-based classification (if available)
    
    Returns
    -------
    List[Dict]
        List of extracted norms with metadata
    """
    norms = []
    
    # Simple pattern-based extraction
    # Look for modal verbs and legal language patterns
    patterns = [
        r'(shall|must|may|should|ought to)\s+([^.!?]+)',
        r'(is|are)\s+(required|prohibited|permitted|allowed|forbidden)\s+to\s+([^.!?]+)',
        r'(no|any)\s+person\s+(shall|may|must)\s+([^.!?]+)',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            norm = {
                'text': match.group(0),
                'modal': match.group(1) if match.lastindex >= 1 else None,
                'type': classify_norm_type(match.group(0)),
                'confidence': 0.7  # Default confidence
            }
            norms.append(norm)
    
    # Use transformer if requested and available
    if use_transformer:
        try:
            classifier = load_transformer_classifier()
            for norm in norms:
                # Enhance with transformer-based classification
                prediction = classifier(norm['text'])
                if prediction:
                    norm['confidence'] = prediction[0].get('score', 0.7)
        except Exception as e:
            print(f"Warning: Transformer classification failed: {e}")
    
    return norms


def classify_norm_type(text: str) -> str:
    """
    Classify the type of legal norm.
    
    Parameters
    ----------
    text : str
        Norm text
        
    Returns
    -------
    str
        Norm type (obligation, prohibition, permission)
    """
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['must', 'shall', 'required', 'obliged']):
        return 'obligation'
    elif any(word in text_lower for word in ['prohibited', 'forbidden', 'not allowed', 'may not']):
        return 'prohibition'
    elif any(word in text_lower for word in ['may', 'permitted', 'allowed', 'can']):
        return 'permission'
    else:
        return 'unknown'


def extract_deontic_operators(text: str) -> List[Dict]:
    """
    Extract deontic operators (obligatory, permitted, forbidden) from text.
    
    Parameters
    ----------
    text : str
        Legal text
        
    Returns
    -------
    List[Dict]
        List of deontic operators with context
    """
    operators = []
    
    # Deontic logic patterns
    obligation_patterns = ['must', 'shall', 'is required to', 'has a duty to', 'obliged to']
    permission_patterns = ['may', 'is permitted to', 'is allowed to', 'can']
    prohibition_patterns = ['must not', 'shall not', 'is prohibited from', 'is forbidden to']
    
    for pattern in obligation_patterns:
        if pattern in text.lower():
            operators.append({
                'type': 'obligation',
                'operator': 'O',
                'pattern': pattern,
                'text': text
            })
    
    for pattern in permission_patterns:
        if pattern in text.lower():
            operators.append({
                'type': 'permission',
                'operator': 'P',
                'pattern': pattern,
                'text': text
            })
    
    for pattern in prohibition_patterns:
        if pattern in text.lower():
            operators.append({
                'type': 'prohibition',
                'operator': 'F',
                'pattern': pattern,
                'text': text
            })
    
    return operators


def extract_conditional_norms(text: str) -> List[Dict]:
    """
    Extract conditional norms (if-then structures) from text.
    
    Parameters
    ----------
    text : str
        Legal text
        
    Returns
    -------
    List[Dict]
        List of conditional norms
    """
    conditionals = []
    
    # Pattern for conditional structures
    if_then_pattern = r'[Ii]f\s+([^,]+),\s+then\s+([^.]+)'
    matches = re.finditer(if_then_pattern, text)
    
    for match in matches:
        conditionals.append({
            'condition': match.group(1).strip(),
            'consequence': match.group(2).strip(),
            'full_text': match.group(0),
            'type': 'conditional'
        })
    
    # Alternative pattern: when... (consequence)
    when_pattern = r'[Ww]hen\s+([^,]+),\s+([^.]+)'
    matches = re.finditer(when_pattern, text)
    
    for match in matches:
        conditionals.append({
            'condition': match.group(1).strip(),
            'consequence': match.group(2).strip(),
            'full_text': match.group(0),
            'type': 'conditional'
        })
    
    return conditionals


def extract_legal_entities(text: str) -> List[Dict]:
    """
    Extract legal entities mentioned in the text.
    
    Parameters
    ----------
    text : str
        Legal text
        
    Returns
    -------
    List[Dict]
        List of legal entities
    """
    entities = []
    
    # Use spaCy for NER if available
    try:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LAW']:
                entities.append({
                    'text': ent.text,
                    'type': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
    except Exception as e:
        print(f"Warning: Entity extraction failed: {e}")
    
    return entities


def analyze_norm_conflicts(norms: List[Dict]) -> List[Dict]:
    """
    Analyze potential conflicts between norms.
    
    Parameters
    ----------
    norms : List[Dict]
        List of extracted norms
        
    Returns
    -------
    List[Dict]
        List of potential conflicts
    """
    conflicts = []
    
    for i, norm1 in enumerate(norms):
        for norm2 in norms[i+1:]:
            # Check if one is obligation and other is prohibition for similar content
            if norm1['type'] == 'obligation' and norm2['type'] == 'prohibition':
                # Simple similarity check (can be enhanced)
                if has_similar_content(norm1['text'], norm2['text']):
                    conflicts.append({
                        'norm1': norm1,
                        'norm2': norm2,
                        'type': 'obligation-prohibition',
                        'severity': 'high'
                    })
    
    return conflicts


def has_similar_content(text1: str, text2: str, threshold: float = 0.5) -> bool:
    """
    Check if two texts have similar content (simple word overlap).
    
    Parameters
    ----------
    text1 : str
        First text
    text2 : str
        Second text
    threshold : float
        Similarity threshold
        
    Returns
    -------
    bool
        True if texts are similar
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return False
    
    similarity = len(intersection) / len(union)
    return similarity >= threshold


# Main extraction function
def extract_all_norms(text: str, use_transformer: bool = False) -> Dict:
    """
    Extract all types of norms from text.
    
    Parameters
    ----------
    text : str
        Legal text to analyze
    use_transformer : bool
        Whether to use transformer models
        
    Returns
    -------
    Dict
        Dictionary containing all extracted norm information
    """
    return {
        'norms': extract_norms(text, use_transformer),
        'deontic_operators': extract_deontic_operators(text),
        'conditionals': extract_conditional_norms(text),
        'entities': extract_legal_entities(text),
        'conflicts': []  # Will be populated after norm extraction
    }
