import re

def extract_fir_info(fir_text):
    info = {
        "location": None,
        "weapon_type": None,
        "injury_type": None,
        "intent": None
    }

    if "kitchen" in fir_text.lower():
        info["location"] = "kitchen"
    if "blunt" in fir_text.lower():
        info["injury_type"] = "blunt force"
    if "stab" in fir_text.lower():
        info["injury_type"] = "stab"
    if "firearm" in fir_text.lower():
        info["weapon_type"] = "firearm"
    if "domestic" in fir_text.lower():
        info["intent"] = "domestic violence"

    return info