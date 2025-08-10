import re
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use lite version to avoid downloads
try:
    from nlp.intent_classifier import IntentClassifier
    print("✅ Lite IntentClassifier imported - no downloads!")
except ImportError:
    print("⚠️ Creating fallback classifier...")
    
    class IntentClassifier:
        def __init__(self):
            print("🔧 Using emergency fallback classifier")
        
        def classify(self, text):
            # Very basic fallback
            text_lower = text.lower()
            if any(word in text_lower for word in ['murder', 'killed', 'dead']):
                return {"predicted_intent": "Homicide", "confidence": 0.7}
            elif any(word in text_lower for word in ['domestic', 'spouse', 'husband', 'wife']):
                return {"predicted_intent": "Domestic Violence", "confidence": 0.7}
            elif any(word in text_lower for word in ['robbed', 'stolen', 'theft']):
                return {"predicted_intent": "Robbery", "confidence": 0.7}
            else:
                return {"predicted_intent": "Unknown", "confidence": 0.5}

try:
    from nlp.rag_retriever import find_similar_firs, load_fir_corpus
    print("✅ RAG retriever imported")
except ImportError as e:
    print(f"⚠️ RAG import failed: {e}")
    def find_similar_firs(text, corpus, top_k=3):
        return ["No similar cases available"]
    def load_fir_corpus(path):
        return []

class SceneAgent:
    """Main crime scene analysis agent"""
    
    def __init__(self):
        print("🤖 Initializing Scene Agent...")
        
        # Initialize intent classifier
        try:
            self.intent_classifier = IntentClassifier()
            print("✅ Intent classifier ready")
        except Exception as e:
            print(f"⚠️ Intent classifier failed: {e}")
            self.intent_classifier = None
        
        # Load corpus
        try:
            self.corpus = load_fir_corpus("data/firs")
            print(f"✅ Loaded {len(self.corpus)} historical cases")
        except Exception as e:
            print(f"⚠️ Corpus loading failed: {e}")
            self.corpus = []
        
        # Crime patterns for extraction
        self.crime_patterns = {
            'weapons': ['knife', 'gun', 'weapon', 'bat', 'stick'],
            'violence': ['fight', 'hit', 'beat', 'attack', 'assault'],
            'locations': ['kitchen', 'bedroom', 'living room', 'bathroom'],
            'evidence': ['blood', 'fingerprint', 'dna', 'witness']
        }
        
        print("✅ Scene Agent initialized")
    
    def analyze(self, fir_text: str, detected_objects: List[str] = None) -> Tuple[Dict, List[str], str]:
        """Main analysis method"""
        if detected_objects is None:
            detected_objects = []
        
        print(f"🔍 Analyzing FIR text ({len(fir_text)} characters)")
        print(f"👁️ Visual objects: {detected_objects}")
        
        # Step 1: Classify crime type
        crime_info = self._classify_crime_type(fir_text)
        
        # Step 2: Extract key information
        extracted_info = self._extract_key_info(fir_text, detected_objects)
        
        # Step 3: Find similar cases
        similar_cases = self._find_similar_cases(fir_text)
        
        # Step 4: Generate reasoning
        reasoning = self._generate_reasoning(crime_info, extracted_info, similar_cases, detected_objects)
        
        # Combine information
        full_info = {
            **crime_info,
            **extracted_info,
            'detected_objects': detected_objects
        }
        
        return full_info, similar_cases, reasoning
    
    def _classify_crime_type(self, text: str) -> Dict:
        """Classify the crime type"""
        if self.intent_classifier:
            try:
                result = self.intent_classifier.classify(text)
                return {
                    'crime_type': result['predicted_intent'],
                    'confidence': result['confidence']
                }
            except Exception as e:
                print(f"⚠️ Classification failed: {e}")
        
        # Fallback classification
        return {
            'crime_type': 'Unknown',
            'confidence': 0.5
        }
    
    def _extract_key_info(self, text: str, objects: List[str]) -> Dict:
        """Extract key information from text"""
        info = {
            'key_phrases': [],
            'potential_evidence': [],
            'risk_factors': []
        }
        
        text_lower = text.lower()
        
        # Extract weapons
        for weapon in self.crime_patterns['weapons']:
            if weapon in text_lower:
                info['potential_evidence'].append(weapon)
                info['risk_factors'].append(f'Weapon mentioned: {weapon}')
        
        # Extract violence indicators
        for violence in self.crime_patterns['violence']:
            if violence in text_lower:
                info['risk_factors'].append(f'Violence indicator: {violence}')
        
        # Add visual objects as evidence
        info['potential_evidence'].extend(objects)
        
        return info
    
    def _find_similar_cases(self, text: str) -> List[str]:
        """Find similar historical cases"""
        if self.corpus:
            try:
                return find_similar_firs(text, self.corpus, top_k=3)
            except Exception as e:
                print(f"⚠️ Similar case search failed: {e}")
        
        return ["No similar cases available"]
    
    def _generate_reasoning(self, crime_info: Dict, extracted_info: Dict, 
                          similar_cases: List[str], objects: List[str]) -> str:
        """Generate AI reasoning"""
        reasoning_parts = []
        
        # Crime classification reasoning
        reasoning_parts.append(f"🎯 Crime Classification: {crime_info['crime_type']}")
        reasoning_parts.append(f"Confidence: {crime_info['confidence']:.2%}")
        
        # Evidence analysis
        if extracted_info['potential_evidence']:
            reasoning_parts.append(f"🔍 Evidence identified: {', '.join(extracted_info['potential_evidence'])}")
        
        # Risk assessment
        if extracted_info['risk_factors']:
            reasoning_parts.append(f"⚠️ Risk factors: {len(extracted_info['risk_factors'])} identified")
        
        # Visual evidence
        if objects:
            reasoning_parts.append(f"👁️ Visual evidence supports analysis: {len(objects)} objects detected")
        
        # Similar cases
        if similar_cases and "No similar cases" not in similar_cases[0]:
            reasoning_parts.append(f"📚 Found {len(similar_cases)} similar historical cases")
        
        return "\n".join(reasoning_parts)

# Test the enhanced Scene Agent
if __name__ == "__main__":
    # Test with enhanced functionality
    agent = SceneAgent()
    
    test_cases = [
        {
            "fir": "Victim found unconscious in kitchen with head injuries. Husband claims she fell down stairs. No witnesses. History of arguments.",
            "objects": ["bat", "chair", "blood_stain"]
        },
        {
            "fir": "Store owner found dead behind counter. Cash register emptied. Back door forced open. Security camera damaged.",
            "objects": ["gun", "cash_register", "broken_glass"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"TEST CASE {i}")
        print(f"{'='*50}")
        
        info, cases, reasoning = agent.analyze(test_case["fir"], test_case["objects"])
        
        print("🔍 EXTRACTED INFO:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print(f"\n📚 SIMILAR CASES: {len(cases)} found")
        
        print(f"\n🧠 REASONING:")
        print(reasoning)


