import os
import json
from pathlib import Path
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from torchvision.ops import nms
from PIL import Image
import cv2
import numpy as np

# Cache directory for detection results
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Crime-relevant objects only (reduces false positives)
CRIME_OBJECTS = {
    "knife", "scissors", "bat", "bottle", "gun", "pistol", "rifle",
    "chair", "table", "bed", "couch", "phone", "laptop", "bag",
    "backpack", "suitcase", "box", "car", "truck", "bicycle"
}

# High-risk objects that need extra attention
HIGH_RISK_OBJECTS = {"knife", "gun", "pistol", "rifle", "bat", "scissors"}

class SceneDetector:
    """Enhanced YOLOS detector with caching and false-positive reduction"""
    
    def __init__(self, model_choice="yolos-small"):
        print("🔍 Initializing Enhanced YOLOS Scene Detector...")
        self.model_choice = model_choice
        
        # Try to load local model first
        model_path = f"models/{model_choice}"
        if os.path.exists(model_path):
            print(f"📁 Loading cached model from {model_path}")
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            self.model = AutoModelForObjectDetection.from_pretrained(model_path)
        else:
            print(f"🌐 Downloading model: {model_choice}")
            model_name = "hustvl/yolos-small" if model_choice == "yolos-small" else model_choice
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForObjectDetection.from_pretrained(model_name)
            
            # Save for future use
            os.makedirs(model_path, exist_ok=True)
            self.processor.save_pretrained(model_path)
            self.model.save_pretrained(model_path)
            print(f"💾 Model cached to {model_path}")
        
        print("✅ Enhanced YOLOS detector ready!")
    
    def analyze_scene_image(self, image_path):
        """Analyze image with caching and false-positive reduction"""
        print(f"🖼️ Analyzing: {Path(image_path).name}")
        
        # Check cache first
        cache_file = os.path.join(CACHE_DIR, f"{Path(image_path).stem}_detection.json")
        if os.path.exists(cache_file):
            print(f"💾 Loading cached results for {Path(image_path).name}")
            try:
                with open(cache_file, "r") as f:
                    cached_result = json.load(f)
                    # Add source image to each object if not present
                    for obj in cached_result.get("objects", []):
                        if "source_image" not in obj:
                            obj["source_image"] = Path(image_path).name
                    return cached_result
            except Exception as e:
                print(f"⚠️ Cache read error: {e}, performing fresh detection")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Run detection with enhanced threshold
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process with higher confidence threshold
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, 
                threshold=0.75,  # Higher threshold to reduce false positives
                target_sizes=target_sizes
            )[0]
            
            # Apply Non-Maximum Suppression to remove duplicate detections
            if len(results["boxes"]) > 0:
                keep = nms(results["boxes"], results["scores"], iou_threshold=0.4)
                results = {k: v[keep] for k, v in results.items()}
            
            # Filter and format objects
            detected_objects = []
            confidence_warnings = []
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                label_name = self.model.config.id2label[label.item()].lower()
                confidence = float(score)
                
                # Only include crime-relevant objects
                if label_name in CRIME_OBJECTS:
                    obj_data = {
                        "object": label_name,
                        "confidence": confidence,
                        "source_image": Path(image_path).name,
                        "detection_method": "yolos_ai",
                        "bbox": [float(x) for x in box.tolist()]
                    }
                    
                    # Add confidence warning for uncertain detections
                    if confidence < 0.85:
                        obj_data["warning"] = "⚠️ Possible false detection - verify manually"
                        confidence_warnings.append(f"{label_name} ({confidence:.2%})")
                    
                    # Mark high-risk objects
                    if label_name in HIGH_RISK_OBJECTS:
                        obj_data["risk_level"] = "HIGH"
                        obj_data["priority"] = "URGENT"
                    
                    detected_objects.append(obj_data)
            
            # Identify suspicious patterns
            suspicious_patterns = self._identify_suspicious_patterns(detected_objects)
            
            # Add confidence warnings to patterns
            if confidence_warnings:
                suspicious_patterns.append(f"⚠️ LOW CONFIDENCE DETECTIONS: {', '.join(confidence_warnings)}")
            
            # Create result
            result = {
                "objects": detected_objects,
                "suspicious_patterns": suspicious_patterns,
                "total_detections": len(detected_objects),
                "image_dimensions": list(image.size),
                "detection_quality": self._assess_detection_quality(detected_objects),
                "cache_timestamp": str(Path(image_path).stat().st_mtime),
                "analysis_timestamp": str(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
            }
            
            # Save to cache
            try:
                with open(cache_file, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"💾 Results cached for future use")
            except Exception as e:
                print(f"⚠️ Cache save error: {e}")
            
            print(f"✅ Detected {len(detected_objects)} crime-relevant objects")
            for obj in detected_objects:
                risk_indicator = " 🚨" if obj.get("risk_level") == "HIGH" else ""
                warning_indicator = " ⚠️" if obj.get("warning") else ""
                print(f"  • {obj['object']} ({obj['confidence']:.2%}){risk_indicator}{warning_indicator}")
            
            return result
            
        except Exception as e:
            print(f"❌ Detection error for {image_path}: {e}")
            # Return fallback with error indication
            return {
                "objects": [{
                    "object": "error_fallback",
                    "confidence": 0.50,
                    "source_image": Path(image_path).name,
                    "warning": "⚠️ Detection failed - manual review required",
                    "error": str(e)
                }],
                "suspicious_patterns": ["🚨 DETECTION ERROR - Manual analysis required"],
                "total_detections": 0,
                "error": str(e)
            }
    
    def _identify_suspicious_patterns(self, objects):
        """Enhanced pattern identification with confidence awareness"""
        patterns = []
        object_names = [obj["object"].lower() for obj in objects]
        high_conf_objects = [obj for obj in objects if obj["confidence"] > 0.85]
        low_conf_objects = [obj for obj in objects if obj["confidence"] < 0.85]
        
        # High-risk weapon detection
        weapons = [obj for obj in objects if obj["object"] in HIGH_RISK_OBJECTS]
        if weapons:
            high_conf_weapons = [w for w in weapons if w["confidence"] > 0.85]
            low_conf_weapons = [w for w in weapons if w["confidence"] < 0.85]
            
            if high_conf_weapons:
                weapon_names = [w["object"] for w in high_conf_weapons]
                patterns.append(f"🔴 HIGH CONFIDENCE WEAPONS: {', '.join(weapon_names)}")
            
            if low_conf_weapons:
                weapon_names = [w["object"] for w in low_conf_weapons]
                patterns.append(f"🟡 POSSIBLE WEAPONS (verify): {', '.join(weapon_names)}")
        
        # Multiple weapons
        if len(weapons) > 1:
            patterns.append(f"⚠️ MULTIPLE POTENTIAL WEAPONS: Escalated threat level")
        
        # Weapon + concealment
        containers = ['bag', 'backpack', 'suitcase', 'box']
        if any(c in object_names for c in containers) and weapons:
            patterns.append("🚨 CONCEALMENT RISK: Weapon + container combination")
        
        # Vehicle + weapon (getaway preparation)
        vehicles = ['car', 'truck', 'bicycle']
        if any(v in object_names for v in vehicles) and weapons:
            patterns.append("🚗 ESCAPE PREPARATION: Vehicle + weapon detected")
        
        # High confidence detection summary
        if len(high_conf_objects) > 3:
            patterns.append(f"✅ {len(high_conf_objects)} high-confidence detections")
        
        # Low confidence warning
        if len(low_conf_objects) > len(high_conf_objects):
            patterns.append(f"⚠️ {len(low_conf_objects)} low-confidence detections need verification")
        
        return patterns
    
    def _assess_detection_quality(self, objects):
        """Assess overall detection quality"""
        if not objects:
            return "no_detections"
        
        avg_confidence = sum(obj["confidence"] for obj in objects) / len(objects)
        high_conf_count = sum(1 for obj in objects if obj["confidence"] > 0.85)
        high_conf_ratio = high_conf_count / len(objects)
        
        if avg_confidence > 0.9 and high_conf_ratio > 0.8:
            return "excellent"
        elif avg_confidence > 0.8 and high_conf_ratio > 0.6:
            return "good"
        elif avg_confidence > 0.7:
            return "fair"
        else:
            return "poor_needs_verification"
    
    def clear_cache(self):
        """Clear detection cache"""
        import shutil
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR, exist_ok=True)
            print("🗑️ Detection cache cleared")

# Utility function for cache management
def clear_detection_cache():
    """Clear all cached detections"""
    detector = SceneDetector()
    detector.clear_cache()

def get_cache_stats():
    """Get cache statistics"""
    if not os.path.exists(CACHE_DIR):
        return {"cached_files": 0, "cache_size": "0 MB"}
    
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
    cache_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in cache_files)
    
    return {
        "cached_files": len(cache_files),
        "cache_size": f"{cache_size / (1024*1024):.2f} MB"
    }

if __name__ == "__main__":
    # Test the enhanced detector
    detector = SceneDetector()
    print("\n🧪 Testing Enhanced Detection...")
    
    # Print cache stats
    stats = get_cache_stats()
    print(f"📊 Cache Stats: {stats['cached_files']} files, {stats['cache_size']}")