# app/streamlit_app.py

import streamlit as st
from pipeline.crime_pipeline import CrimePipeline, CrimeScene
import os
import tempfile
import datetime

# Initialize pipeline
pipeline = CrimePipeline()

st.set_page_config(page_title="🔍 Crime Scene Analyzer", layout="wide")

st.sidebar.title("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Min. Confidence for Classification", 0.0, 1.0, 0.7, 0.01)
top_k_cases = st.sidebar.slider("Similar Cases to Fetch", 1, 5, 3)

st.title("🚨 AI Crime Scene Analyzer")
st.markdown("Upload an **FIR** (as text) and optionally a **scene image** for analysis.")

fir_text = st.text_area("📄 Paste FIR Text", height=200, placeholder="e.g. Victim found in kitchen, head trauma...")
uploaded_image = st.file_uploader("📸 Upload Crime Scene Image", type=["jpg", "png"], accept_multiple_files=False)
case_id = st.text_input("🆔 Case ID (optional)", value=f"CASE_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")

if st.button("🔍 Analyze Crime Scene") and fir_text:
    image_path = None

    # Save uploaded image to temp dir if provided
    if uploaded_image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            tmp_img.write(uploaded_image.read())
            image_path = tmp_img.name

    scene = CrimeScene(
        fir_text=fir_text,
        image_path=image_path,
        detected_objects=None,
        location=None,
        timestamp=None
    )

    with st.spinner("Analyzing..."):
        result = pipeline.analyze_crime_scene(scene)

    # Display results
    st.success("✅ Analysis Complete")

    st.subheader("🧠 Predicted Crime Type")
    st.write(result.extracted_info.get("intent", "Unknown"))

    st.subheader("📊 Confidence Score")
    st.progress(result.confidence_score)

    st.subheader("📍 Timeline")
    st.markdown("- FIR parsed and classified.\n- Visual objects integrated.\n- Reasoning and similar cases fetched.")

    st.subheader("🧾 Extracted Information")
    st.json(result.extracted_info)

    st.subheader("🧠 AI Reasoning")
    st.markdown(result.reasoning or "No reasoning available.")

    st.subheader("📚 Similar Cases")
    for i, case in enumerate(result.similar_cases):
        st.markdown(f"**{i+1}.** {case}")

    if result.visual_analysis:
        st.subheader("👁️ Visual Evidence")
        st.write(f"Objects Detected: {result.visual_analysis.get('total_objects')}")
        suspicious = result.visual_analysis.get('suspicious_patterns', [])
        if suspicious:
            st.write(f"Suspicious Patterns: {', '.join(suspicious)}")
else:
    if st.button("🔍 Analyze Crime Scene"):
        st.warning("Please enter FIR text to begin analysis.")
