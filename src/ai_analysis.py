from utils import genai, GENAI_AVAILABLE, DeepFace, DEEPFACE_AVAILABLE
from PIL import Image
import numpy as np

API_KEY = "AIzaSyD8YutZfmK98-Tbg8BDvhXEaI2yBjvTAjg"

if GENAI_AVAILABLE:
    try:
        genai.configure(api_key=API_KEY)
    except:
        GENAI_AVAILABLE = False

def analyze_with_gemini(image):
    """Analyze image with Gemini API"""
    if not GENAI_AVAILABLE:
        return "AI analysis unavailable (module not installed)"
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([
            "Analyze this surveillance image. Describe: 1) Number of people 2) Their activities 3) Any suspicious behavior 4) Risk level (Low/Medium/High). Be concise.",
            image
        ])
        return response.text if response.text else "No analysis generated"
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            return "AI model temporarily unavailable (404 error)"
        return f"Analysis failed: {error_msg[:100]}"

def analyze_person_attributes(frame_crop):
    """Analyze person attributes with DeepFace"""
    if not DEEPFACE_AVAILABLE or frame_crop is None or frame_crop.size == 0:
        return {"age": "N/A", "gender": "N/A", "emotion": "N/A"}
    
    try:
        result = DeepFace.analyze(
            frame_crop,
            actions=['age', 'gender', 'emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        if isinstance(result, list):
            result = result[0]
        
        return {
            "age": str(int(result.get("age", 0))) if result.get("age") else "N/A",
            "gender": result.get("dominant_gender", "N/A"),
            "emotion": result.get("dominant_emotion", "N/A")
        }
    except:
        return {"age": "N/A", "gender": "N/A", "emotion": "N/A"}

def rule_based_analysis(person_count, idle_count, in_restricted_zone):
    """Fallback rule-based analysis"""
    risk = "Low"
    analysis = []
    
    if person_count == 0:
        analysis.append("No persons detected")
    elif person_count == 1:
        analysis.append("Single person present")
        if idle_count > 0:
            analysis.append("Person appears idle")
            risk = "Low"
    else:
        analysis.append(f"{person_count} persons detected")
        if person_count > 5:
            analysis.append("Crowded area")
            risk = "Medium"
    
    if in_restricted_zone:
        analysis.append("Person in restricted zone")
        risk = "High"
    
    if idle_count > 2:
        analysis.append(f"{idle_count} persons idle/loitering")
        risk = "Medium"
    
    return f"Risk: {risk}. " + ". ".join(analysis)
