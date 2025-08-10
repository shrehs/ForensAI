# app/streamlit_app.py

import streamlit as st
import sys
import os
import tempfile
import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Now import from your actual modules
from scene_pipeline import ScenePipeline, CrimeSceneInput, AnalysisOutput

def init_pipeline(model_choice):
    """Initialize the pipeline with error handling"""
    try:
        return ScenePipeline(model_choice=model_choice)
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        return None

def main():
    st.set_page_config(
        page_title="🔍 Crime Scene Analyzer", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar settings
    st.sidebar.title("⚙️ Settings")
    confidence_threshold = st.sidebar.slider("Min. Confidence for Classification", 0.0, 1.0, 0.7, 0.01)
    top_k_cases = st.sidebar.slider("Similar Cases to Fetch", 1, 5, 3)
    
    # Add model selection
    model_choice = st.sidebar.selectbox(
        "Detection Mode",
        ["lite", "ultra-lite"],
        index=0,
        help="Choose detection mode (all options are lightweight)"
    )

    # Move this INSIDE the main() function, in the sidebar section
    st.sidebar.subheader("🛠️ System Management")
    
    try:
        from vision.detector import get_cache_stats, clear_detection_cache
        cache_stats = get_cache_stats()
        st.sidebar.write(f"📊 **Cache Stats:**")
        st.sidebar.write(f"Files: {cache_stats['cached_files']}")
        st.sidebar.write(f"Size: {cache_stats['cache_size']}")
        
        if st.sidebar.button("🗑️ Clear Cache"):
            clear_detection_cache()
            st.sidebar.success("Cache cleared!")
            st.rerun()
    except ImportError:
        st.sidebar.info("Cache management not available")

    # Main interface
    st.title("🚨 AI Crime Scene Analyzer")
    st.markdown("Upload an **FIR** (as text) and optionally **scene images** for comprehensive analysis.")

    # Initialize pipeline
    if 'pipeline' not in st.session_state:
        with st.spinner("Initializing AI models..."):
            st.session_state.pipeline = init_pipeline(model_choice)

    if st.session_state.pipeline is None:
        st.error("❌ Pipeline initialization failed. Please check your setup.")
        return

    pipeline = st.session_state.pipeline

    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📄 Case Information")

        # Timeline guidance message
        st.info(
            "💡 **Tip:** For better analysis, please mention the timeline of events in your FIR. "
            "Include specific times, dates, and sequence words (e.g., 'first', 'then', 'after', 'before', 'around 8:30 PM')."
        )

        # Check if sample case was loaded
        default_fir = ""
        if 'sample_fir' in st.session_state:
            default_fir = st.session_state.sample_fir
            del st.session_state.sample_fir

        fir_text = st.text_area(
            "FIR Text",
            height=150,
            value=default_fir,
            placeholder=(
                "e.g. Victim found unconscious in kitchen with head trauma around 9:15 PM. "
                "Earlier at 8:30 PM, neighbors heard loud argument. Husband claims she fell down stairs. "
                "Baseball bat found nearby."
            )
        )
        
        # NEW: Timeline Entry Section
        st.subheader("⏰ Event Timeline")
        st.caption("Add key events with timestamps to improve timeline analysis")
        
        # Initialize timeline events in session state if not exists
        if 'timeline_events' not in st.session_state:
            st.session_state.timeline_events = []
            
        # Display existing timeline events
        if st.session_state.timeline_events:
            st.write("📅 Recorded Events:")
            for i, event in enumerate(st.session_state.timeline_events):
                col_e1, col_e2, col_e3 = st.columns([1, 2, 0.5])
                with col_e1:
                    st.text(event['time'])
                with col_e2:
                    st.text(event['description'])
                with col_e3:
                    if st.button("🗑️", key=f"del_{i}"):
                        st.session_state.timeline_events.pop(i)
                        st.rerun()
        
        # Add new timeline event
        col_t1, col_t2, col_t3 = st.columns([1, 2, 1])
        with col_t1:
            event_time = st.text_input("Time", placeholder="e.g. 8:30 PM", key="new_time")
        with col_t2:
            event_desc = st.text_input("Event Description", placeholder="e.g. Neighbors heard argument", key="new_desc")
        with col_t3:
            if st.button("➕ Add Event"):
                if event_time and event_desc:
                    st.session_state.timeline_events.append({
                        'time': event_time,
                        'description': event_desc
                    })
                    # Instead of trying to clear the widget value, just rerun
                    st.rerun()
        
        # NEW: Add timeline events to FIR text when analyzing
        if st.session_state.timeline_events:
            timeline_text = "\n\nTimeline of events:\n"
            for event in st.session_state.timeline_events:
                timeline_text += f"- {event['time']}: {event['description']}\n"
            
            # This will be appended to FIR text during analysis
            full_fir_text = fir_text + timeline_text
        else:
            full_fir_text = fir_text
        
        # Rest of your existing inputs
        case_id = st.text_input(
            "🆔 Case ID (optional)", 
            value=f"CASE_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        location = st.text_input("📍 Location (optional)", placeholder="e.g. 123 Main Street")

    with col2:
        st.subheader("📸 Visual Evidence")
        uploaded_images = st.file_uploader(
            "Upload Crime Scene Images", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True
        )
        
        if uploaded_images:
            st.write(f"📷 {len(uploaded_images)} image(s) uploaded")
            
            # Show thumbnails
            for img in uploaded_images[:3]:  # Show first 3
                st.image(img, width=100)

    # SINGLE ANALYSIS BUTTON - FIXED!
    if st.button("🔍 Analyze Crime Scene", type="primary"):
        if not full_fir_text:
            st.warning("⚠️ Please enter FIR text to begin analysis.")
        else:
            image_paths = []

            # Save uploaded images to temp files
            if uploaded_images:
                temp_dir = tempfile.mkdtemp()
                for i, uploaded_image in enumerate(uploaded_images):
                    temp_path = os.path.join(temp_dir, f"scene_{i}.jpg")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_image.read())
                    image_paths.append(temp_path)

            # Create scene input with full_fir_text
            scene_input = CrimeSceneInput(
                fir_text=full_fir_text,  # Use the combined text with timeline
                image_paths=image_paths,
                case_id=case_id,
                location=location,
                timestamp=datetime.datetime.now()
            )

            # Perform analysis
            with st.spinner("🔍 Analyzing crime scene..."):
                try:
                    analysis = pipeline.analyze_scene(scene_input)
                    
                    # Store in session state for download
                    st.session_state.last_analysis = analysis
                    
                    # Display results
                    display_results(analysis, pipeline)
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.exception(e)

def display_results(analysis: AnalysisOutput, pipeline):
    """Display the analysis results"""
    
    st.success("✅ Analysis Complete!")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Crime Type", 
            analysis.crime_type,
            help="AI-predicted crime classification"
        )
    
    with col2:
        st.metric(
            "Confidence", 
            f"{analysis.confidence_score:.1%}",
            help="Model confidence in classification"
        )
    
    with col3:
        st.metric(
            "Timeline Events", 
            len(analysis.timeline),
            help="Number of events extracted"
        )
    
    with col4:
        risk_level = analysis.risk_assessment.get('overall_risk', 'Unknown').upper()
        risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🟣"}.get(risk_level, "⚪")
        st.metric(
            "Risk Level", 
            f"{risk_color} {risk_level}",
            help="Overall risk assessment"
        )

    # Tabbed results
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⏰ Timeline", "👁️ Visual Evidence", "⚠️ Risk Assessment", 
        "📚 Similar Cases", "🧠 AI Reasoning"
    ])
    
    with tab1:
        st.subheader("📅 Reconstructed Timeline")
        
        if analysis.timeline:
            for i, event in enumerate(analysis.timeline, 1):
                with st.expander(f"Event {i}: {event.event_type.title()}", expanded=i<=3):
                    st.write(f"**Description:** {event.description}")
                    st.write(f"**Source:** {event.source}")
                    st.write(f"**Confidence:** {event.confidence:.2%}")
                    
                    if event.evidence:
                        st.write(f"**Evidence:** {', '.join(event.evidence)}")
                    
                    if event.timestamp:
                        st.write(f"**Timestamp:** {event.timestamp.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.info("No timeline events extracted")

    with tab2:
        st.subheader("🔍 Visual Evidence Analysis")
        
        if analysis.visual_evidence.get('objects'):
            st.write("**Detected Objects:**")
            
            # Create columns for objects
            objects = analysis.visual_evidence['objects']
            cols = st.columns(min(4, len(objects)))
            
            for i, obj in enumerate(objects):
                with cols[i % 4]:
                    # Handle both string and dict objects
                    if isinstance(obj, dict):
                        obj_name = obj.get('object', str(obj))
                        confidence = obj.get('confidence', 0)
                        st.write(f"• {obj_name} ({confidence:.1%})")
                    else:
                        st.write(f"• {obj}")
            
            # Show suspicious patterns if any
            detailed_analysis = analysis.visual_evidence.get('detailed_analysis', {})
            if detailed_analysis:
                st.write("**Detailed Analysis:**")
                for img_path, img_analysis in detailed_analysis.items():
                    if img_analysis.get('suspicious_patterns'):
                        st.warning(f"🚨 Suspicious patterns: {', '.join(img_analysis['suspicious_patterns'])}")
        else:
            st.info("No visual evidence processed")

    with tab3:
        st.subheader("⚠️ Risk Assessment")
        
        risk_data = analysis.risk_assessment
        
        # Overall risk with color coding
        risk_level = risk_data.get('overall_risk', 'Unknown').upper()
        risk_colors = {
            "LOW": "🟢 Low Risk",
            "MEDIUM": "🟡 Medium Risk", 
            "HIGH": "🔴 High Risk",
            "CRITICAL": "🟣 Critical Risk"
        }
        
        st.markdown(f"**Overall Risk Level:** {risk_colors.get(risk_level, f'⚪ {risk_level}')}")
        
        # Specific risks
        if risk_data.get('specific_risks'):
            st.write("**Identified Risks:**")
            for risk in risk_data['specific_risks']:
                st.write(f"• {risk}")
        
        # Urgency factors
        if risk_data.get('urgency_factors'):
            st.write("**🚨 Urgent Factors:**")
            for factor in risk_data['urgency_factors']:
                st.error(f"⚡ {factor}")
        
        # Protective measures
        if risk_data.get('protective_measures'):
            st.write("**🛡️ Recommended Protective Measures:**")
            for measure in risk_data['protective_measures']:
                st.info(f"🔒 {measure}")

    with tab4:
        st.subheader("📚 Similar Historical Cases")
        
        if analysis.similar_cases and "Error" not in str(analysis.similar_cases[0]):
            for i, case in enumerate(analysis.similar_cases, 1):
                # If your similar_cases are dicts with 'intent' and 'text'
                if isinstance(case, dict):
                    with st.expander(f"Case {i}: {case.get('intent', 'Unknown')}", expanded=(i==1)):
                        st.markdown(f"**Intent:** `{case.get('intent', 'Unknown')}`")
                        if case.get('location'):
                            st.markdown(f"**Location:** `{case['location']}`")
                        st.markdown(f"**Narrative:**")
                        st.write(case.get('text', ''))
                else:
                    # If it's just text, fallback
                    with st.expander(f"Case {i}", expanded=(i==1)):
                        st.write(case)
        else:
            st.info("No similar cases found or retrieval error occurred")

    with tab5:
        st.subheader("🧠 AI Analysis & Reasoning")
        st.markdown(analysis.generated_reasoning)
        
        # Additional technical details
        with st.expander("🔧 Technical Details"):
            st.json({
                "processing_timestamp": analysis.processing_timestamp.isoformat(),
                "case_id": analysis.case_id,
                "text_analysis_details": analysis.text_analysis
            })

    # Download section
    st.subheader("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Generate report
        report_text = pipeline.generate_report(analysis)
        st.download_button(
            label="📄 Download Report (TXT)",
            data=report_text,
            file_name=f"{analysis.case_id}_report.txt",
            mime="text/plain"
        )
    
    with col2:
        # Generate JSON
        import json
        analysis_dict = {
            "case_id": analysis.case_id,
            "crime_type": analysis.crime_type,
            "confidence_score": analysis.confidence_score,
            "timeline": [
                {
                    "description": event.description,
                    "event_type": event.event_type,
                    "confidence": event.confidence,
                    "evidence": event.evidence
                }
                for event in analysis.timeline
            ],
            "risk_assessment": analysis.risk_assessment,
            "recommendations": analysis.recommendations
        }
        
        st.download_button(
            label="📊 Download Data (JSON)",
            data=json.dumps(analysis_dict, indent=2),
            file_name=f"{analysis.case_id}_data.json",
            mime="application/json"
        )

# Sample cases for demo
def show_sample_cases():
    st.sidebar.subheader("📋 Sample Cases")
    
    sample_cases = {
        "Domestic Violence": "Victim found unconscious in kitchen with head trauma. Husband claims she fell down stairs. Neighbors heard loud argument earlier. Baseball bat found nearby.",
        
        "Armed Robbery": "Store clerk found shot behind counter. Cash register emptied. Back door forced open. Security camera destroyed. Suspect fled on foot.",
        
        "Suspicious Death": "Elderly man found dead in apartment. Door was unlocked. No signs of struggle. Empty pill bottle on nightstand. Suicide note found."
    }
    
    for case_name, case_text in sample_cases.items():
        if st.sidebar.button(f"Load: {case_name}"):
            st.session_state.sample_fir = case_text
            st.rerun()  # Force refresh to load the sample

if __name__ == "__main__":
    show_sample_cases()
    main()
