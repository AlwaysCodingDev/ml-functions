import re
from typing import List, Set, Dict
from collections import defaultdict

class FastKeywordMatcher:
    def __init__(self):
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'or', 'but', 'not', 'this', 'have'
        }
    
    def kmp_search(self, text: str, pattern: str) -> bool:
        """KMP algorithm for pattern matching"""
        if not pattern:
            return True
        if not text:
            return False
            
        # Build failure function
        def build_failure_function(pattern: str) -> List[int]:
            failure = [0] * len(pattern)
            j = 0
            for i in range(1, len(pattern)):
                while j > 0 and pattern[i] != pattern[j]:
                    j = failure[j - 1]
                if pattern[i] == pattern[j]:
                    j += 1
                failure[i] = j
            return failure
        
        failure = build_failure_function(pattern)
        i = j = 0
        
        while i < len(text):
            if text[i] == pattern[j]:
                i += 1
                j += 1
            if j == len(pattern):
                return True
            elif i < len(text) and text[i] != pattern[j]:
                if j > 0:
                    j = failure[j - 1]
                else:
                    i += 1
        return False
    
    def preprocess_text(self, text: str) -> str:
        """Normalize text for better matching"""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def extract_ngrams(self, text: str, n: int = 2) -> Set[str]:
        """Extract n-grams for partial matching"""
        words = text.split()
        ngrams = set()
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i + n])
            if not all(word in self.stop_words for word in words[i:i + n]):
                ngrams.add(ngram)
        return ngrams
    
    def calculate_keyword_weights(self, keywords: List[str]) -> Dict[str, float]:
        """Assign weights to keywords based on importance"""
        weights = {}
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Longer keywords get higher weight
            base_weight = min(len(keyword_lower) / 10, 2.0)
            
            # Technical terms get bonus weight
            if any(tech in keyword_lower for tech in ['python', 'java', 'sql', 'aws', 'docker', 'kubernetes']):
                base_weight *= 1.5
            
            # Multi-word phrases get bonus weight
            if len(keyword_lower.split()) > 1:
                base_weight *= 1.3
            
            weights[keyword_lower] = max(base_weight, 1.0)
        
        return weights
    
    def fuzzy_match_score(self, keyword: str, text: str) -> float:
        """Calculate fuzzy matching score using various techniques"""
        keyword = keyword.lower()
        text = text.lower()
        
        # Exact match (highest score)
        if self.kmp_search(text, keyword):
            return 1.0
        
        # Word boundary match
        word_pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(word_pattern, text):
            return 1.0
        
        # Partial word matches for compound keywords
        keyword_words = keyword.split()
        if len(keyword_words) > 1:
            matches = sum(1 for word in keyword_words 
                         if word not in self.stop_words and self.kmp_search(text, word))
            partial_score = matches / len(keyword_words)
            if partial_score > 0.5:  # At least half the words match
                return partial_score * 0.8
        
        # N-gram similarity for close matches
        text_ngrams = self.extract_ngrams(text, 2)
        keyword_ngrams = self.extract_ngrams(keyword, 2)
        
        if keyword_ngrams:
            ngram_matches = len(keyword_ngrams.intersection(text_ngrams))
            ngram_score = ngram_matches / len(keyword_ngrams)
            if ngram_score > 0.3:
                return ngram_score * 0.6
        
        # Character-level similarity for typos/variations
        def jaccard_similarity(s1: str, s2: str) -> float:
            set1, set2 = set(s1), set(s2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0
        
        char_sim = jaccard_similarity(keyword, text)
        if char_sim > 0.7:
            return char_sim * 0.4
        
        return 0.0

def get_keywords_similarity_score(resume_text: str, keywords: List[str]) -> float:
    """Calculate similarity based on job keywords using fast algorithms"""
    if not resume_text or not keywords:
        return 0.0
    
    matcher = FastKeywordMatcher()
    
    # Preprocess resume text
    processed_text = matcher.preprocess_text(resume_text)
    
    # Calculate keyword weights
    keyword_weights = matcher.calculate_keyword_weights(keywords)
    
    # Calculate matches
    total_weight = 0.0
    matched_weight = 0.0
    match_details = []
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        weight = keyword_weights.get(keyword_lower, 1.0)
        total_weight += weight
        
        # Calculate fuzzy match score for this keyword
        match_score = matcher.fuzzy_match_score(keyword_lower, processed_text)
        matched_weight += match_score * weight
        
        if match_score > 0:
            match_details.append(f"{keyword}: {match_score:.2f}")
    
    # Calculate final similarity score
    if total_weight == 0:
        return 0.0
    
    base_score = (matched_weight / total_weight) * 100
    
    # Apply bonus for high match density
    unique_matches = len([detail for detail in match_details if float(detail.split(': ')[1]) > 0.5])
    total_keywords = len(keywords)
    
    if total_keywords > 0:
        match_ratio = unique_matches / total_keywords
        if match_ratio > 0.8:
            base_score *= 1.1  # 10% bonus for high match rate
        elif match_ratio > 0.6:
            base_score *= 1.05  # 5% bonus for good match rate
    
    # Ensure score is within bounds
    final_score = min(100.0, max(0.0, base_score))
    
    # Optional: Print debug information
    # print(f"Matched keywords: {match_details}")
    # print(f"Final score: {final_score:.1f}")
    
    return final_score