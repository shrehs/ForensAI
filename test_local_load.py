from vision.detector import SceneDetector
import os

def test_no_download():
    """Test that model loads without any downloads"""
    
    # Disable internet access for transformers (optional safety check)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    try:
        print("🧪 Testing local model loading...")
        detector = SceneDetector(model_choice="yolos-small")
        print("✅ Model loaded successfully from local files!")
        
        # Test with dummy analysis if you have a test image
        test_image = "data/images/sample.jpg"  # Replace with actual path
        if os.path.exists(test_image):
            result = detector.analyze_scene_image(test_image)
            print(f"🔍 Test analysis: {result['total_detections']} objects detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Local loading failed: {e}")
        return False

if __name__ == "__main__":
    test_no_download()