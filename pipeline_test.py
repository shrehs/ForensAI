# Pipeline Integration Test
import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_full_pipeline():
    try:
        # 1. Test RAG Retriever
        from nlp.rag_retriever import find_similar_firs
        print("[OK] RAG Retriever imported successfully")
        
        # 2. Test Vision Detector  
        from vision.detector import SceneDetector
        detector = SceneDetector()
        print("[OK] Scene Detector initialized")
        
        # 3. Test Scene Agent
        from agent.scene_agent import SceneAgent
        agent = SceneAgent()
        print("[OK] Scene Agent initialized")
        
        # 4. End-to-end test
        fir_text = "Test incident report"
        detected_objects = ["chair", "table"]
        
        info, cases, reasoning = agent.analyze(fir_text, detected_objects)
        print("[OK] Full pipeline executed successfully")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Pipeline error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_full_pipeline()
    if success:
        print("\n[SUCCESS] All pipeline components working!")
    else:
        print("\n[FAILED] Pipeline has issues - check imports and dependencies")
