import re
from typing import Dict, List

class IntentClassifier:
    """Ultra-lightweight intent classifier - NO DOWNLOADS"""
    
    def __init__(self):
        print("🪶 Initializing Lite Intent Classifier (no downloads)...")
        
        # Pattern-based classification - no models needed!
        self.crime_patterns = {
            'Homicide': [
                'murder', 'killed', 'dead body', 'corpse', 'fatal', 'death', 
                'deceased', 'shot dead', 'stabbed', 'strangled', 'beaten to death',
                'homicide', 'slain', 'victim found dead'
            ],
            'Domestic Violence': [
                'domestic', 'spouse', 'husband', 'wife', 'family', 'home violence', 
                'partner', 'boyfriend', 'girlfriend', 'marriage', 'relationship',
                'family dispute', 'marital', 'domestic abuse', 'restraining order'
            ],
            'Robbery': [
                'robbed', 'robbery', 'stolen', 'theft', 'burglar', 'break-in', 
                'loot', 'cash', 'money', 'valuables', 'store', 'bank',
                'heist', 'armed robbery', 'burglary', 'shoplifting'
            ],
            'Sexual Assault': [
                'rape', 'sexual assault', 'molestation', 'harassment', 
                'inappropriate touching', 'molest', 'sexual abuse'
            ],
            'Drug Related': [
                'drugs', 'narcotics', 'cocaine', 'heroin', 'marijuana', 
                'substance', 'dealer', 'trafficking', 'possession',
                'overdose', 'drug dealing', 'meth', 'cannabis'
            ],
            'Suicide': [
                'suicide', 'self-harm', 'hanging', 'overdose', 'jumped', 
                'pills', 'suicide note', 'depression', 'self-inflicted'
            ],
            'Accidental Death': [
                'accident', 'accidental', 'mishap', 'fell', 'slipped', 
                'unintentional', 'mistake', 'car accident', 'drowning'
            ],
            'Missing Person': [
                'missing', 'disappeared', 'vanished', 'not found', 
                'whereabouts unknown', 'last seen', 'abducted', 'kidnapped'
            ],
            'Fraud': [
                'fraud', 'scam', 'embezzlement', 'forgery', 'identity theft',
                'credit card fraud', 'ponzi scheme', 'fake', 'counterfeit'
            ],
            'Assault': [
                'assault', 'battery', 'fight', 'attacked', 'beaten up',
                'physical altercation', 'punched', 'kicked', 'hit'
            ]
        }
        
        self.candidate_labels = list(self.crime_patterns.keys())
        print(f"✅ Pattern-based classifier ready! ({len(self.candidate_labels)} crime types)")
    
    def classify(self, text: str) -> Dict:
        """Classify using keyword patterns - INSTANT, NO DOWNLOADS"""
        text_lower = text.lower()
        
        # Score each crime type
        scores = {}
        best_score = 0
        best_crime = "Unknown"
        best_keywords = []
        
        for crime_type, keywords in self.crime_patterns.items():
            matched_keywords = []
            score = 0
            
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            # Normalize score (0-1)
            confidence = score / len(keywords) if keywords else 0
            scores[crime_type] = confidence
            
            # Track best match
            if confidence > best_score:
                best_score = confidence
                best_crime = crime_type
                best_keywords = matched_keywords
        
        # If no good match found
        if best_score < 0.05:  # Very low threshold
            return {
                "predicted_intent": "Unknown",
                "confidence": 0.0,
                "matched_keywords": [],
                "all_scores": scores,
                "method": "pattern_matching"
            }
        
        return {
            "predicted_intent": best_crime,
            "confidence": best_score,
            "matched_keywords": best_keywords,
            "all_scores": scores,
            "method": "pattern_matching"
        }

# Create global instance for easy import
intent_classifier = IntentClassifier()

def classify_intent(text: str) -> str:
    """Quick classification function"""
    result = intent_classifier.classify(text)
    return result['predicted_intent']