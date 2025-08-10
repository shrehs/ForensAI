import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.scene_agent import SceneAgent

fir_text = "Victim found in kitchen, blunt force trauma, no weapon recovered. Possible domestic situation."
detected_objects = ["bat", "chair", "refrigerator"]

agent = SceneAgent()
info, similar_cases, reasoning = agent.analyze(fir_text, detected_objects)

print("🔍 Extracted Info:", info)
print("\n📚 Similar Cases:\n", "\n---\n".join(similar_cases))
print("\n🧠 SceneAgent Reasoning:\n", reasoning)
