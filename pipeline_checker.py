import os
import sys
from pathlib import Path

class PipelineChecker:
    def __init__(self, project_root="d:\\Crime\\Proj"):
        self.project_root = Path(project_root)
        self.issues = []
        self.recommendations = []
    
    def check_dependencies(self):
        """Check if all required packages are available"""
        required_packages = [
            'transformers', 'torch', 'sklearn', 'numpy', 
            'cv2', 'PIL', 'sentence_transformers', 'faiss'
        ]
        
        missing = []
        for package in required_packages:
            try:
                if package == 'cv2':
                    import cv2
                elif package == 'PIL':
                    from PIL import Image
                elif package == 'faiss':
                    try:
                        import faiss
                    except ImportError:
                        try:
                            import faiss_cpu as faiss
                        except ImportError:
                            missing.append('faiss-cpu')
                            continue
                else:
                    __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            self.issues.append(f"Missing packages: {', '.join(missing)}")
            if 'cv2' in missing:
                self.recommendations.append("Install: pip install opencv-python")
            if 'PIL' in missing:
                self.recommendations.append("Install: pip install Pillow")
            if 'faiss-cpu' in missing:
                self.recommendations.append("Install: pip install faiss-cpu")
            
            other_missing = [p for p in missing if p not in ['cv2', 'PIL', 'faiss-cpu']]
            if other_missing:
                self.recommendations.append(f"Install: pip install {' '.join(other_missing)}")
    
    def check_file_structure(self):
        """Check if all required files exist"""
        required_files = [
            "agent/scene_agent.py",
            "nlp/rag_retriever.py", 
            "vision/detector.py",
            "app/test_run.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.issues.append(f"Missing files: {', '.join(missing_files)}")
    
    def check_data_availability(self):
        """Check if data directories exist and create them"""
        data_dirs = ["data/firs", "data/images", "models"]
        
        created_dirs = []
        for dir_path in data_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                try:
                    full_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(dir_path)
                except Exception as e:
                    self.issues.append(f"Cannot create directory {dir_path}: {str(e)}")
        
        if created_dirs:
            self.recommendations.append(f"Created directories: {', '.join(created_dirs)}")
    
    def generate_pipeline_test(self):
        """Generate comprehensive pipeline test"""
        return """# Pipeline Integration Test
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
        print("\\n[SUCCESS] All pipeline components working!")
    else:
        print("\\n[FAILED] Pipeline has issues - check imports and dependencies")
"""
    
    def create_sample_fir_data(self):
        """Create sample FIR data for testing"""
        firs_dir = self.project_root / "data" / "firs"
        if firs_dir.exists() and not any(firs_dir.iterdir()):
            sample_firs = [
                "Robbery at convenience store. Armed suspect with handgun. Cash register emptied. CCTV footage available.",
                "Domestic violence incident. Victim found with bruises. Broken furniture in living room. Suspect fled scene.",
                "Drug possession case. Suspect found with cocaine and marijuana. Digital scale and plastic bags recovered.",
                "Assault case. Victim attacked with baseball bat. Multiple witnesses. Suspect known to victim.",
                "Burglary report. House ransacked. Electronics and jewelry missing. No signs of forced entry."
            ]
            
            for i, fir_content in enumerate(sample_firs, 1):
                fir_file = firs_dir / f"sample_fir_{i}.txt"
                with open(fir_file, 'w', encoding='utf-8') as f:
                    f.write(fir_content)
            
            self.recommendations.append(f"Created {len(sample_firs)} sample FIR files for testing")
    
    def run_full_check(self):
        """Run complete pipeline check"""
        print("Checking Crime Analysis Pipeline...")
        print()
        
        self.check_dependencies()
        self.check_file_structure() 
        self.check_data_availability()
        self.create_sample_fir_data()
        
        # Generate report
        print("PIPELINE ANALYSIS REPORT")
        print("=" * 40)
        
        if self.issues:
            print()
            print("ISSUES FOUND:")
            for issue in self.issues:
                print(f"  - {issue}")
        else:
            print()
            print("[SUCCESS] No critical issues found!")
        
        if self.recommendations:
            print()
            print("ACTIONS TAKEN/RECOMMENDATIONS:")
            for rec in self.recommendations:
                print(f"  - {rec}")
        
        # Generate test file
        test_file = self.project_root / "pipeline_test.py"
        try:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(self.generate_pipeline_test())
            
            print()
            print(f"Generated pipeline test: {test_file}")
            print("Run: python pipeline_test.py")
            
        except Exception as e:
            print(f"Error creating test file: {e}")

# Run the check
if __name__ == "__main__":
    checker = PipelineChecker()
    checker.run_full_check()