ecgnew/ecg.jpeg
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import google.generativeai as genai
from datetime import datetime
import io
import base64
import time
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from PIL import Image as PILImage
from tensorflow.keras.preprocessing import image as keras_image
from huggingface_hub import hf_hub_download
import sys

# --------------------
# ADVANCED CONFIG
# --------------------
st.set_page_config(
    page_title="ECG Health Analyzer", 
    page_icon="🫀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .risk-high { border-left-color: #e74c3c !important; }
    .risk-medium { border-left-color: #f39c12 !important; }
    .risk-low { border-left-color: #27ae60 !important; }
    .profile-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        margin-bottom: 2rem;
    }
    .step-indicator {
        display: flex;
        justify-content: center;
        margin-bottom: 2rem;
    }
    .step {
        padding: 10px 20px;
        margin: 0 10px;
        border-radius: 20px;
        background: #e9ecef;
        color: #6c757d;
    }
    .step.active {
        background: #667eea;
        color: white;
    }
    .step.completed {
        background: #28a745;
        color: white;
    }
    .get-started-btn {
        background-color: #28a745;
        color: white;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: background-color 0.3s, transform 0.2s;
        margin-top: 2rem;
    }
    .get-started-btn:hover {
        background-color: #218838;
        transform: scale(1.05);
    }
    .get-started-btn:active {
        transform: scale(0.95);
    }
    .landing-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 80vh;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --------------------
# CONFIGURATION & CONSTANTS
# --------------------
GEMINI_API_KEY = "AIzaSyAYilbLvhYzckWlcBqIvTKcdIwdN5DNj4k"

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Class labels with detailed descriptions (matching model order)
CLASS_NAMES = ['Abnormal', 'History_of_MI', 'MI', 'Normal']

CLASS_INFO = {
 "Abnormal": {
    "color": "#f39c12", 
    "risk_level": "Medium",
    "description": "Your heart rhythm is not fully normal. This could mean your heartbeat is irregular or has some unusual patterns. It’s not an emergency, but you should see a doctor soon to find out the cause and prevent future problems."
},
"History_of_MI": {
    "color": "#d35400",
    "risk_level": "Medium-High",
    "description": "Signs show that you may have had a heart attack in the past. People with a past heart attack have a higher chance of having another one. Regular check-ups, healthy lifestyle habits, and following your doctor’s advice are very important."
},
"MI": {
    "color": "#e74c3c",
    "risk_level": "High", 
    "description": "This reading suggests you may be having a heart attack right now. This is a medical emergency. Call emergency services immediately to get urgent treatment and reduce the risk of serious damage or death."
},
"Normal": {
    "color": "#27ae60",
    "risk_level": "Low",
    "description": "Your heart rhythm looks normal and healthy. Keep up healthy habits like eating well, exercising regularly, and going for check-ups to maintain good heart health."
}

}

# --------------------
# SESSION STATE INITIALIZATION
# --------------------
if 'profile_completed' not in st.session_state:
    st.session_state.profile_completed = False
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'analysis_completed' not in st.session_state:
    st.session_state.analysis_completed = False
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'show_landing' not in st.session_state:
    st.session_state.show_landing = True

# --------------------
# ADVANCED MODEL LOADING
# --------------------
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="gk784/ecg-risk-model",
            filename="ecg_risk_model.h5"
        )
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

# --------------------
# ADVANCED IMAGE PROCESSING
# --------------------
def preprocess_image(uploaded_file):
    """Preprocess image exactly like the model training approach"""
    model = load_model()
    if model is None:
        target_size = (180, 180)
    else:
        _, height, width, channels = model.input_shape
        target_size = (height, width)
    
    # Load image using Keras preprocessing
    img = keras_image.load_img(io.BytesIO(uploaded_file.getvalue()), target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def analyze_image_quality(img):
    """Analyze uploaded image quality"""
    img_array = np.array(img)
    brightness = np.mean(img_array)
    contrast = np.std(img_array)
    quality_score = min(100, (contrast / 50) * 100)
    return {
        "brightness": brightness,
        "contrast": contrast,
        "quality_score": quality_score,
        "resolution": img.size
    }

# --------------------
# AI SUGGESTIONS WITH GEMINI
# --------------------
def get_advanced_suggestions(label, patient_info=None):
    """Get comprehensive health suggestions using Gemini AI"""
    patient_context = ""
    if patient_info:
        patient_context = f"""
        Patient Information:
        - Name: {patient_info.get('name', 'Not specified')}
        - Age: {patient_info.get('age', 'Not specified')}
        - Gender: {patient_info.get('gender', 'Not specified')}
        - Medical History: {patient_info.get('history', 'None reported')}
        - Current Medications: {patient_info.get('medications', 'None reported')}
        - Current Symptoms: {', '.join(patient_info.get('symptoms', ['None']))}
        """
    
    prompt = f"""
    As a cardiac health specialist, analyze this ECG classification: '{label}'.
    
    {patient_context}
    
    Provide a comprehensive health report including:
    
    1. **Immediate Actions** (if any urgent steps needed)
    2. **Dietary Recommendations** (specific foods and nutrients)
    3. **Exercise Guidelines** (appropriate activities and intensity)
    4. **Lifestyle Modifications** (sleep, stress management, habits)
    5. **Monitoring Suggestions** (what to watch for, follow-up timeline)
    6. **Risk Factors** (what could worsen or improve the condition)
    
    Format the response with clear headings and bullet points. Keep medical language accessible to general public while maintaining accuracy.
    """
    
    try:
        model_gemini = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Unable to generate suggestions: {e}"

# --------------------
# STEP INDICATOR
# --------------------
def show_step_indicator():
    """Display step progress indicator"""
    step1_class = "completed" if st.session_state.profile_completed else "active" if st.session_state.current_step == 1 else ""
    step2_class = "completed" if st.session_state.analysis_completed else "active" if st.session_state.current_step == 2 and st.session_state.profile_completed else ""
    step3_class = "active" if st.session_state.current_step == 3 and st.session_state.analysis_completed else ""
    
    st.markdown(f"""
    <div class="step-indicator">
        <div class="step {step1_class}">1. Profile Setup</div>
        <div class="step {step2_class}">2. Upload & Analyze</div>
        <div class="step {step3_class}">3. View Results</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------
# BACKGROUND SETTER
# --------------------
def set_background(image_path=None):
    st.markdown(
    """
    <style>
    .stApp {
    background: linear-gradient(135deg, #e3f2fd, #90a4ae);
    
    
    }
    </style>
    """,
    unsafe_allow_html=True
    )

# --------------------
# LANDING PAGE
# --------------------
def show_landing_page():
    """Display landing page with centered Get Started button"""
    # Hide Streamlit default top bar and menu
    st.markdown(
        """
        <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            div.stButton > button {
                background: linear-gradient(90deg, #27ae60, #2ecc71);
                color: white;
                padding: 12px 30px;
                font-size: 1rem;
                border: none;
                border-radius: 25px;
                cursor: pointer;
            }
        
        </style>
        """,
        unsafe_allow_html=True
    )


    # Heading
    st.markdown(
        '<p style="color: white; text-align: center; font-size:44px; font-weight: bold;">🫀 AI-POWERED ECG ANALYSER</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="color: white; text-align: center; font-size: 18px;">AI-Powered Cardiac Health Analysis with Personalized Insights</p>',
        unsafe_allow_html=True
    )

    # Use columns to center button
    col1, col2, col3 = st.columns([8, 3, 7])
    with col2:
        if st.button("Get Started", key="get_started",type="primary"):
            st.session_state.show_landing = False
            st.session_state.current_step = 1
            st.rerun()

# --------------------
# PROFILE COLLECTION
# --------------------
def collect_patient_profile():
    """Collect comprehensive patient profile"""
    st.subheader("👤 Patient Profile Setup")
    st.write("Please provide your information for personalized ECG analysis:")
    
    with st.form("patient_profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name *", placeholder="Enter your full name")
            age = st.number_input("Age *", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender *", ["Male", "Female", "Other"])
        with col2:
            weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0, step=0.1)
        
        st.write("**Medical Information:**")
        history = st.text_area("Medical History", placeholder="Previous heart conditions, surgeries, family history, etc.", height=100)
        medications = st.text_area("Current Medications", placeholder="List current medications and dosages", height=100)
        symptoms = st.multiselect("Current Symptoms", ["Chest Pain", "Shortness of Breath", "Palpitations", "Dizziness", "Fatigue", "Nausea", "Sweating", "None"])
        emergency_contact = st.text_input("Emergency Contact", placeholder="Name and phone number")
        
        submitted = st.form_submit_button("✅ Complete Profile Setup", use_container_width=True)
        
        if submitted:
            if name and age and gender:
                st.session_state.patient_data = {
                    'name': name,
                    'age': age,
                    'gender': gender,
                    'weight': weight,
                    'history': history,
                    'medications': medications,
                    'symptoms': symptoms,
                    'emergency_contact': emergency_contact,
                    'profile_created': datetime.now().isoformat()
                }
                st.session_state.profile_completed = True
                st.session_state.current_step = 2
                st.success("✅ Profile completed successfully! You can now upload your ECG image.")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Please fill in all required fields (marked with *)")

# --------------------
# ECG UPLOAD AND ANALYSIS
# --------------------
def ecg_upload_analysis():
    """Handle ECG image upload and analysis"""
    st.subheader("📤 ECG Image Upload & Analysis")
    
    with st.expander("👤 Patient Profile Summary", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {st.session_state.patient_data.get('name', 'N/A')}")
            st.write(f"**Age:** {st.session_state.patient_data.get('age', 'N/A')}")
            st.write(f"**Gender:** {st.session_state.patient_data.get('gender', 'N/A')}")
        with col2:
            st.write(f"**Weight:** {st.session_state.patient_data.get('weight', 'N/A')} kg")
        
        if st.button("📝 Edit Profile", use_container_width=True):
            st.session_state.profile_completed = False
            st.session_state.current_step = 1
            st.rerun()
    
    uploaded_file = st.file_uploader(
        "📤 Upload ECG Image", 
        type=["png", "jpg", "jpeg"],
        help="Upload a clear ECG image for analysis. Supported formats: PNG, JPG, JPEG"
    )
    
    show_preprocessing = st.checkbox("👁️ Show Processing Steps")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        if show_preprocessing:
            st.subheader("📷 Image Processing Steps")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Original Image**")
                st.image(image, use_container_width=True)
            with col2:
                model = load_model()
                if model:
                    _, height, width, channels = model.input_shape
                    target_size = (height, width)
                else:
                    target_size = (180, 180)
                processed = image.resize(target_size)
                st.write(f"**Resized to {target_size}**")
                st.image(processed, use_container_width=True)
            with col3:
                img_array = np.array(processed) / 255.0
                st.write("**Normalized [0-1]**")
                st.image(img_array, use_container_width=True)
        else:
            st.image(image, caption="Uploaded ECG Image", use_container_width=True)
        
        quality_metrics = analyze_image_quality(image)
        
        st.subheader("📊 Image Quality Assessment")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            quality_color = "green" if quality_metrics['quality_score'] > 70 else "orange" if quality_metrics['quality_score'] > 40 else "red"
            st.metric("Quality Score", f"{quality_metrics['quality_score']:.1f}%", 
                     help=f"Image quality assessment based on contrast and clarity")
        with col2:
            st.metric("Brightness", f"{quality_metrics['brightness']:.1f}")
        with col3:
            st.metric("Contrast", f"{quality_metrics['contrast']:.1f}")
        with col4:
            st.metric("Resolution", f"{quality_metrics['resolution'][0]}×{quality_metrics['resolution'][1]}")
        
        if quality_metrics['quality_score'] < 50:
            st.warning("⚠️ Image quality is below optimal. Consider uploading a clearer image for better analysis accuracy.")
        elif quality_metrics['quality_score'] < 70:
            st.info("ℹ️ Image quality is acceptable. Analysis will proceed with good accuracy.")
        else:
            st.success("✅ Excellent image quality detected. Analysis will have high accuracy.")
        
        if st.button("🔍 Analyze ECG", use_container_width=True, type="primary"):
            analyze_ecg(uploaded_file, quality_metrics)

def analyze_ecg(uploaded_file, quality_metrics):
    """Perform ECG analysis and store results"""
    model = load_model()
    if model is None:
        st.error("❌ Model not available. Please check the model path.")
        return
    
    with st.spinner("🔄 Analyzing ECG pattern... This may take a moment."):
        img_array = preprocess_image(uploaded_file)
        predictions = model.predict(img_array, verbose=0)
        pred_class_idx = np.argmax(predictions, axis=1)[0]
        label = CLASS_NAMES[pred_class_idx]
    
    st.session_state.analysis_results = {
        'diagnosis': label,
        'quality_metrics': quality_metrics,
        'timestamp': datetime.now().isoformat()
    }
    st.session_state.analysis_completed = True
    st.session_state.current_step = 3
    st.success("✅ ECG Analysis Complete!")
    time.sleep(1)
    st.rerun()

# --------------------
# RESULTS DISPLAY
# --------------------
def show_analysis_results():
    """Display comprehensive analysis results"""
    if 'analysis_results' not in st.session_state:
        st.error("❌ No analysis results found.")
        return
    
    results = st.session_state.analysis_results
    label = results['diagnosis']
    quality_metrics = results['quality_metrics']
    
    st.subheader("🩺 ECG Analysis Results")
    
    risk_class = f"risk-{CLASS_INFO[label]['risk_level'].lower().replace('-', '')}"
    st.markdown(f"""
    <div class="metric-card {risk_class}">
        <h3>🩺 Primary Diagnosis: {label}</h3>
        <p><strong>Risk Level:</strong> {CLASS_INFO[label]['risk_level']}</p>
        <p><strong>Analysis Date:</strong> {datetime.fromisoformat(results['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>{CLASS_INFO[label]['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Start New Analysis", use_container_width=True):
            st.session_state.profile_completed = False
            st.session_state.analysis_completed = False
            st.session_state.current_step = 1
            st.session_state.patient_data = {}
            st.session_state.analysis_results = {}
            st.session_state.show_landing = True
            st.rerun()
    
    st.subheader("🤖 Personalized Health Recommendations")
    with st.spinner("Generating personalized recommendations based on your profile..."):
        suggestions = get_advanced_suggestions(label, st.session_state.patient_data)
    
    st.markdown(suggestions)
    
    st.subheader("📥 Download Report")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📄 Download PDF Report", use_container_width=True, type="primary"):
            pdf_data = generate_pdf_report(label, suggestions, quality_metrics)
            st.download_button(
                label="📥 Download PDF",
                data=pdf_data,
                file_name=f"ECG_Report_{st.session_state.patient_data.get('name', 'Patient').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
    with col2:
        if st.button("🖼️ Download JPEG Report", use_container_width=True, type="primary"):
            jpeg_data = generate_jpeg_report(label, suggestions, quality_metrics)
            st.download_button(
                label="📥 Download JPEG",
                data=jpeg_data,
                file_name=f"ECG_Report_{st.session_state.patient_data.get('name', 'Patient').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpeg",
                mime="image/jpeg"
            )
    with col3:
        if st.button("🖼️ Download PNG Report", use_container_width=True, type="primary"):
            png_data = generate_png_report(label, suggestions, quality_metrics)
            st.download_button(
                label="📥 Download PNG",
                data=png_data,
                file_name=f"ECG_Report_{st.session_state.patient_data.get('name', 'Patient').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )

def generate_pdf_report(diagnosis, suggestions, quality_metrics):
    """Generate PDF report using reportlab"""
    patient_data = st.session_state.patient_data
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph("ECG Analysis Report", styles['Title']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Patient Information", styles['Heading2']))
    story.append(Paragraph(f"Name: {patient_data.get('name', 'Not specified')}", styles['Normal']))
    story.append(Paragraph(f"Age: {patient_data.get('age', 'Not specified')} years", styles['Normal']))
    story.append(Paragraph(f"Gender: {patient_data.get('gender', 'Not specified')}", styles['Normal']))
    story.append(Paragraph(f"Weight: {patient_data.get('weight', 'Not specified')} kg", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Medical History", styles['Heading2']))
    story.append(Paragraph(f"Previous Conditions: {patient_data.get('history', 'None reported')}", styles['Normal']))
    story.append(Paragraph(f"Current Medications: {patient_data.get('medications', 'None reported')}", styles['Normal']))
    story.append(Paragraph(f"Current Symptoms: {', '.join(patient_data.get('symptoms', ['None']))}", styles['Normal']))
    story.append(Paragraph(f"Emergency Contact: {patient_data.get('emergency_contact', 'Not provided')}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("ECG Analysis Results", styles['Heading2']))
    story.append(Paragraph(f"Primary Diagnosis: {diagnosis}", styles['Normal']))
    story.append(Paragraph(f"Risk Level: {CLASS_INFO[diagnosis]['risk_level']}", styles['Normal']))
    story.append(Paragraph(f"Clinical Description: {CLASS_INFO[diagnosis]['description']}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Image Quality Assessment", styles['Heading2']))
    story.append(Paragraph(f"Quality Score: {quality_metrics['quality_score']:.1f}%", styles['Normal']))
    story.append(Paragraph(f"Resolution: {quality_metrics['resolution'][0]}×{quality_metrics['resolution'][1]} pixels", styles['Normal']))
    story.append(Paragraph(f"Brightness Index: {quality_metrics['brightness']:.1f}", styles['Normal']))
    story.append(Paragraph(f"Contrast Index: {quality_metrics['contrast']:.1f}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("AI-Generated Health Recommendations", styles['Heading2']))
    for line in suggestions.split('\n'):
        if line.startswith('#'):
            story.append(Paragraph(line.lstrip('# '), styles['Heading3']))
        elif line.startswith('*'):
            story.append(Paragraph(line.lstrip('* '), styles['Normal']))
        else:
            story.append(Paragraph(line, styles['Normal']))
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Important Disclaimer: This report is generated by AI technology for screening and educational purposes only. It should not replace professional medical diagnosis, treatment, or advice. Always consult with qualified healthcare professionals for medical decisions.", styles['Normal']))
    story.append(Paragraph("Emergency: If you are experiencing chest pain, shortness of breath, or other serious symptoms, seek immediate medical attention or call emergency services.", styles['Normal']))
    
    doc.build(story)
    return buffer.getvalue()

def generate_jpeg_report(diagnosis, suggestions, quality_metrics):
    """Generate JPEG report (simplified text-based image)"""
    patient_data = st.session_state.patient_data
    img_width, img_height = 800, 1200
    img = PILImage.new('RGB', (img_width, img_height), color='white')
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    y_position = 20
    line_spacing = 30
    
    def add_text(text, size=20, bold=False):
        nonlocal y_position
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except:
            font = ImageFont.load_default()
        draw.text((20, y_position), text, font=font, fill='black')
        y_position += line_spacing
    
    add_text("ECG Analysis Report", size=30)
    y_position += 10
    add_text(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y_position += 10
    
    add_text("Patient Information", size=25)
    add_text(f"Name: {patient_data.get('name', 'Not specified')}")
    add_text(f"Age: {patient_data.get('age', 'Not specified')} years")
    add_text(f"Gender: {patient_data.get('gender', 'Not specified')}")
    add_text(f"Weight: {patient_data.get('weight', 'Not specified')} kg")
    y_position += 10
    
    add_text("Medical History", size=25)
    add_text(f"Previous Conditions: {patient_data.get('history', 'None reported')}")
    add_text(f"Current Medications: {patient_data.get('medications', 'None reported')}")
    add_text(f"Current Symptoms: {', '.join(patient_data.get('symptoms', ['None']))}")
    add_text(f"Emergency Contact: {patient_data.get('emergency_contact', 'Not provided')}")
    y_position += 10
    
    add_text("ECG Analysis Results", size=25)
    add_text(f"Primary Diagnosis: {diagnosis}")
    add_text(f"Risk Level: {CLASS_INFO[diagnosis]['risk_level']}")
    add_text(f"Clinical Description: {CLASS_INFO[diagnosis]['description']}")
    y_position += 10
    
    add_text("Image Quality Assessment", size=25)
    add_text(f"Quality Score: {quality_metrics['quality_score']:.1f}%")
    add_text(f"Resolution: {quality_metrics['resolution'][0]}×{quality_metrics['resolution'][1]} pixels")
    add_text(f"Brightness Index: {quality_metrics['brightness']:.1f}")
    add_text(f"Contrast Index: {quality_metrics['contrast']:.1f}")
    y_position += 10
    
    add_text("AI-Generated Health Recommendations", size=25)
    for line in suggestions.split('\n')[:10]:  # Limit to prevent overflow
        add_text(line)
    
    add_text("Important Disclaimer: This report is generated by AI technology for screening and educational purposes only.", size=15)
    add_text("Emergency: If you are experiencing chest pain, shortness of breath, or other serious symptoms, seek immediate medical attention.", size=15)
    
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()

def generate_png_report(diagnosis, suggestions, quality_metrics):
    """Generate PNG report (simplified text-based image)"""
    patient_data = st.session_state.patient_data
    img_width, img_height = 800, 1200
    img = PILImage.new('RGB', (img_width, img_height), color='white')
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    y_position = 20
    line_spacing = 30
    
    def add_text(text, size=20, bold=False):
        nonlocal y_position
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except:
            font = ImageFont.load_default()
        draw.text((20, y_position), text, font=font, fill='black')
        y_position += line_spacing
    
    add_text("ECG Analysis Report", size=30)
    y_position += 10
    add_text(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y_position += 10
    
    add_text("Patient Information", size=25)
    add_text(f"Name: {patient_data.get('name', 'Not specified')}")
    add_text(f"Age: {patient_data.get('age', 'Not specified')} years")
    add_text(f"Gender: {patient_data.get('gender', 'Not specified')}")
    add_text(f"Weight: {patient_data.get('weight', 'Not specified')} kg")
    y_position += 10
    
    add_text("Medical History", size=25)
    add_text(f"Previous Conditions: {patient_data.get('history', 'None reported')}")
    add_text(f"Current Medications: {patient_data.get('medications', 'None reported')}")
    add_text(f"Current Symptoms: {', '.join(patient_data.get('symptoms', ['None']))}")
    add_text(f"Emergency Contact: {patient_data.get('emergency_contact', 'Not provided')}")
    y_position += 10
    
    add_text("ECG Analysis Results", size=25)
    add_text(f"Primary Diagnosis: {diagnosis}")
    add_text(f"Risk Level: {CLASS_INFO[diagnosis]['risk_level']}")
    add_text(f"Clinical Description: {CLASS_INFO[diagnosis]['description']}")
    y_position += 10
    
    add_text("Image Quality Assessment", size=25)
    add_text(f"Quality Score: {quality_metrics['quality_score']:.1f}%")
    add_text(f"Resolution: {quality_metrics['resolution'][0]}×{quality_metrics['resolution'][1]} pixels")
    add_text(f"Brightness Index: {quality_metrics['brightness']:.1f}")
    add_text(f"Contrast Index: {quality_metrics['contrast']:.1f}")
    y_position += 10
    
    add_text("AI-Generated Health Recommendations", size=25)
    for line in suggestions.split('\n')[:10]:  # Limit to prevent overflow
        add_text(line)
    
    add_text("Important Disclaimer: This report is generated by AI technology for screening and educational purposes only.", size=15)
    add_text("Emergency: If you are experiencing chest pain, shortness of breath, or other serious symptoms, seek immediate medical attention.", size=15)
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

# --------------------
# MAIN APPLICATION
# --------------------
def main():
    if st.session_state.show_landing:
        show_landing_page()
    else:
        set_background()  # Reset background to none
        st.markdown("""
        <div class="main-header">
            <h1 style="color: white; margin: 0;">🫀 AI-POWERED ECG ANALYSER</h1>
            <p style="color: white; margin: 10px 0 0 0; font-size: 18px;">
                AI-Powered Cardiac Health Assessment with Personalized Insights
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        show_step_indicator()
        
        if not st.session_state.profile_completed:
            collect_patient_profile()
        elif not st.session_state.analysis_completed:
            ecg_upload_analysis()
        else:
            show_analysis_results()

# --------------------
# RUN APPLICATION
# --------------------
if __name__ == "__main__":
    main()
