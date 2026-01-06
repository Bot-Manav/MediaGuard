"""
MediaGuard - AI-Powered Deepfake Detection & Media Protection System
Streamlit Web Application
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Any
import traceback

from modules.media_upload import MediaUploadHandler
from modules.ai_analysis import AIAnalysisEngine
from modules.fingerprint_generator import FingerprintGenerator
from modules.blockchain_layer import BlockchainTraceability
from modules.verification import VerificationEngine
from modules.dashboard import ResultDashboard
from modules.fake_news_detection import FakeNewsDetectionEngine
from modules.trust_aggregator import TrustAggregator
from modules.content_safety_engine import ContentSafetyEngine

# Page configuration
st.set_page_config(
    page_title="MediaGuard - Media Protection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional, calm design
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .safe-box {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
    }
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
    }
    .danger-box {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
    }
    h1 {
        color: #1f2937;
    }
    h2 {
        color: #374151;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    if 'content_safety_result' not in st.session_state:
        st.session_state.content_safety_result = None

    """Initialize session state variables"""
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'fingerprint_data' not in st.session_state:
        st.session_state.fingerprint_data = None
    if 'blockchain_record' not in st.session_state:
        st.session_state.blockchain_record = None
    if 'protected_fingerprints' not in st.session_state:
        st.session_state.protected_fingerprints = []
    if 'fake_news_result' not in st.session_state:
        st.session_state.fake_news_result = None
    if 'trust_aggregation' not in st.session_state:
        st.session_state.trust_aggregation = None


def display_multi_modal_results(analysis_result, fake_news_result, trust_aggregation,
                                fingerprint_data, blockchain_record, generate_protection):
    """Display comprehensive multi-modal analysis results"""
    
    st.markdown("---")
    st.markdown("## 🧠 Multi-Modal Trust Analysis")
    
    # Overall Trust Risk (from aggregator)
    if trust_aggregation:
        overall_risk = trust_aggregation.get('overall_trust_risk', 0)
        threat_category = trust_aggregation.get('threat_category', 'Unknown')
        confidence = trust_aggregation.get('confidence', 0)
        
        # Display overall risk with color coding
        if overall_risk >= 70:
            risk_color = "🔴"
            risk_box_class = "danger-box"
        elif overall_risk >= 50:
            risk_color = "🟡"
            risk_box_class = "warning-box"
        else:
            risk_color = "🟢"
            risk_box_class = "safe-box"
        
        st.markdown(f"""
        <div class="result-box {risk_box_class}">
            <h3>{risk_color} Overall Trust Risk: {overall_risk:.1f}%</h3>
            <p><strong>Threat Category:</strong> {threat_category}</p>
            <p><strong>Confidence:</strong> {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        recommendations = trust_aggregation.get('recommendations', [])
        if recommendations:
            st.markdown("### 💡 Recommendations")
            for rec in recommendations:
                st.markdown(f"- {rec}")
    
    # Module Results in Columns
    st.markdown("---")
    st.markdown("### 📊 Module Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🖼️ Image Analysis")
        if analysis_result:
            classification = analysis_result.get('classification', 'unknown')
            confidence = analysis_result.get('confidence', 0)
            deepfake_score = analysis_result.get('details', {}).get('deepfake_score', 0)
            sensitive_score = analysis_result.get('details', {}).get('sensitive_score', 0)
            
            if classification == 'real_safe':
                st.success(f"✅ Real & Safe ({confidence:.1f}%)")
            elif classification == 'deepfake':
                st.error(f"🔴 Deepfake Detected ({confidence:.1f}%)")
            elif classification == 'sensitive':
                st.warning(f"🟡 Sensitive Content ({confidence:.1f}%)")
            else:
                st.info(f"❓ Unknown ({confidence:.1f}%)")
            
            st.metric("Deepfake Risk", f"{deepfake_score:.1f}%")
            st.metric("Sensitive Content", f"{sensitive_score:.1f}%")
        else:
            st.info("Image analysis not performed")
    
    with col2:
        st.markdown("#### 📰 Fake News Analysis")
        if fake_news_result:
            fake_news_likelihood = fake_news_result.get('fake_news_likelihood', 0)
            credibility_score = fake_news_result.get('credibility_score', 0)
            uncertain = fake_news_result.get('uncertain', 0)
            
            if fake_news_likelihood > 60:
                st.error(f"🔴 High Fake News Risk ({fake_news_likelihood:.1f}%)")
            elif fake_news_likelihood > 40:
                st.warning(f"🟡 Moderate Risk ({fake_news_likelihood:.1f}%)")
            else:
                st.success(f"🟢 Low Risk ({fake_news_likelihood:.1f}%)")
            
            st.metric("Fake News Likelihood", f"{fake_news_likelihood:.1f}%")
            st.metric("Credibility Score", f"{credibility_score:.1f}%")
            st.metric("Uncertainty", f"{uncertain:.1f}%")
            
            # Show signals if available
            explanation = fake_news_result.get('explanation', {})
            signals = explanation.get('signals', [])
            if signals:
                with st.expander("View Detection Signals"):
                    for signal in signals[:5]:
                        st.markdown(f"- {signal}")
        else:
            st.info("Fake news analysis not performed")

    # Microsoft Azure Content Safety display (if available)
    try:
        cs_present = False
        if trust_aggregation and trust_aggregation.get('module_results', {}).get('sensitive'):
            cs_present = bool(st.session_state.get('content_safety_result'))

        if cs_present:
            cs = st.session_state.get('content_safety_result') or {}
            st.markdown("---")
            st.markdown("### 🛡️ Microsoft AI – Azure Content Safety")
            risk_score = cs.get('risk_score', 0.0)
            max_severity = cs.get('max_severity', 0)
            st.markdown(f"**Risk Score:** {risk_score:.1f}%  |  **Max Severity:** {max_severity}")

            categories = cs.get('categories', {}) or {}
            if categories:
                for cat, meta in categories.items():
                    severity = meta.get('severity', 0)
                    confidence = meta.get('confidence', 0.0)
                    st.markdown(f"- **{cat}**: Severity {severity}, Confidence {confidence:.1f}%")
            else:
                st.info("No content categories returned by Azure Content Safety")
    except Exception:
        # Never allow display errors to break the UI
        pass
    
    # Detailed Breakdown
    if trust_aggregation:
        st.markdown("---")
        st.markdown("### 🔍 Detailed Breakdown")
        with st.expander("View Module Aggregation Details"):
            st.json(trust_aggregation.get('module_results', {}))
            st.markdown(f"**Explanation:** {trust_aggregation.get('explanation', 'N/A')}")
    
    # Fingerprint Information
    if fingerprint_data:
        st.markdown("---")
        st.markdown("### 🔐 Fingerprint Information")
        col_fp1, col_fp2 = st.columns(2)
        
        with col_fp1:
            st.subheader("Cryptographic Hash")
            st.code(fingerprint_data.get('sha256', 'N/A')[:64] + "...", language=None)
        
        with col_fp2:
            st.subheader("Perceptual Hash")
            st.code(fingerprint_data.get('perceptual_hash', 'N/A'), language=None)
    
    # Blockchain Record
    if blockchain_record:
        st.markdown("---")
        st.markdown("### ⛓️ Blockchain Record")
        st.success(f"✅ Registered on blockchain (Block #{blockchain_record.get('block_number', 'N/A')})")
    
    # Protection Status
    if generate_protection:
        st.markdown("---")
        st.success("🛡️ **Protection Fingerprint Generated Successfully**")


def main():
    """Main application entry point"""
    initialize_session_state()
    
    # Header
    st.title("🛡️ MediaGuard")
    st.markdown("### Multi-Modal Threat Analysis Platform")
    st.markdown("*Independent analysis of media authenticity and information credibility*")
    st.markdown("---")
    
    # Sidebar - User personas
    with st.sidebar:
        st.header("👥 User Personas")
        persona = st.radio(
            "Select your role:",
            ["General User", "Victim / Concerned User", "Platform / Authority"],
            help="Choose your role to customize the experience"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **MediaGuard** - Multi-Modal Threat Analysis:
        - 🖼️ Image Analysis (Deepfake + Authenticity)
        - 📰 Fake News Detection (Text + Context)
        - 🧠 Trust & Risk Aggregation
        - 🔐 Secure Fingerprinting
        - ⛓️ Blockchain Traceability
        
        **Independent Modules:**
        Each module analyzes independently, then results are combined for comprehensive assessment.
        
        **Privacy First:**
        - No long-term storage
        - Temporary processing only
        - User consent required
        """)
        
        st.markdown("---")
        st.markdown("### 🔒 Privacy & Ethics")
        st.markdown("""
        ✅ Data minimization  
        ✅ Temporary processing  
        ✅ User consent  
        ✅ No surveillance
        """)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "🔍 Verify Content", "📊 Dashboard"])
    
    with tab1:
        render_upload_analyze_tab(persona)
    
    with tab2:
        render_verification_tab()
    
    with tab3:
        render_dashboard_tab()


def render_upload_analyze_tab(persona: str):
    """Render the upload and analysis tab"""
    st.header("Upload & Analyze Media")
    
    if persona == "General User":
        st.info("👤 **General User Mode**: Upload media to check authenticity and detect deepfakes.")
    elif persona == "Victim / Concerned User":
        st.warning("🛡️ **Protection Mode**: Upload sensitive media to generate protection fingerprints.")
    else:
        st.info("🏛️ **Authority Mode**: View system statistics and receive hash warnings.")
    
    st.markdown("---")
    
    # Text input for fake news analysis (can be used independently)
    st.subheader("📰 Enter Text for Fake News Analysis")
    st.markdown("**You can analyze text with or without an image.** Enter news headline, caption, or any text claim below:")
    
    col_text1, col_text2 = st.columns(2)
    with col_text1:
        news_headline = st.text_input(
            "📰 News Headline / Title",
            placeholder="Example: 'Scientists discover shocking secret...'",
            help="Enter the headline or title you want to analyze for misinformation",
            key="headline_input"
        )
    with col_text2:
        caption_description = st.text_area(
            "📝 Caption / Description / Claim",
            placeholder="Example: 'Breaking news! This one trick will change your life forever. Doctors hate it!'",
            help="Enter caption, description, or any text claim to analyze",
            height=120,
            key="caption_input"
        )
    
    # Show example
    with st.expander("💡 Example Text to Test"):
        st.markdown("""
        **Try these examples to see fake news detection:**
        
        **High Risk Example:**
        - Headline: "SHOCKING: Scientists reveal secret that doctors don't want you to know!"
        - Caption: "This one trick will change everything. You won't believe what happens next!"
        
        **Low Risk Example:**
        - Headline: "New study published in Nature journal shows promising results"
        - Caption: "According to peer-reviewed research, the findings suggest..."
        """)
    
    st.markdown("---")
    
    # File upload (optional)
    uploaded_file = st.file_uploader(
        "🖼️ Upload Image (Optional)",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Optional: Upload an image to analyze for deepfakes. You can analyze text without an image."
    )
    
    if uploaded_file is not None or (news_headline or caption_description):
        # Display uploaded image if available
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📷 Uploaded Media")
                st.image(uploaded_file, use_container_width=True)
                st.caption(f"File: {uploaded_file.name} | Size: {uploaded_file.size / 1024:.2f} KB")
            
            with col2:
                st.subheader("⚙️ Image Analysis Options")
                analyze_deepfake = st.checkbox("Detect Deepfakes", value=True)
                detect_sensitive = st.checkbox("Detect Sensitive Content", value=True)
                generate_protection = st.checkbox("Generate Protection Fingerprint", 
                                                value=(persona == "Victim / Concerned User"))
                register_blockchain = st.checkbox("Register on Blockchain", 
                                                value=(persona == "Victim / Concerned User"))
        else:
            # No image uploaded, so disable image analysis options
            analyze_deepfake = False
            detect_sensitive = False
            generate_protection = False
            register_blockchain = False
            st.info("ℹ️ No image uploaded. You can still analyze text for fake news detection.")
        
        # Fake news analysis option (always available if text is provided)
        st.markdown("---")
        st.subheader("⚙️ Analysis Options")
        analyze_fake_news = st.checkbox(
            "📰 Detect Fake News in Text", 
            value=bool(news_headline or caption_description),
            help="Analyze the entered text for misinformation indicators"
        )
        
        # Show what will be analyzed
        if news_headline or caption_description:
            st.success(f"✅ Text ready for analysis: {len(news_headline or '') + len(caption_description or '')} characters")
        if uploaded_file is not None:
            st.success("✅ Image ready for analysis")
        
        # Analyze button
        if st.button("🔍 Multi-Modal Analysis", type="primary", use_container_width=True):
            with st.spinner("Processing analysis..."):
                try:
                    # Module 1: Secure Media Upload & Intake (if image provided)
                    upload_handler = None
                    media_data = None
                    fingerprint_data = None
                    
                    if uploaded_file is not None:
                        upload_handler = MediaUploadHandler()
                        media_data = upload_handler.process_upload(uploaded_file)
                        
                        if media_data is None:
                            st.error("❌ Failed to process uploaded file. Please try again.")
                            return
                        
                        # Module 3: Fingerprint Generation (only if image provided)
                        fingerprint_gen = FingerprintGenerator()
                        fingerprint_data = fingerprint_gen.generate(media_data)
                    
                    # Module 2: AI Image Analysis (Independent) - only if image provided
                    analysis_result = None
                    if uploaded_file is not None and (analyze_deepfake or detect_sensitive):
                        analysis_engine = AIAnalysisEngine()
                        analysis_result = analysis_engine.analyze(
                            media_data,
                            check_deepfake=analyze_deepfake,
                            check_sensitive=detect_sensitive
                        )
                    
                    # NEW: Module - Fake News Detection (Independent) - works with or without image
                    fake_news_result = None
                    if analyze_fake_news:
                        fake_news_engine = FakeNewsDetectionEngine()
                        texts_to_analyze = []
                        if news_headline:
                            texts_to_analyze.append(news_headline)
                        if caption_description:
                            texts_to_analyze.append(caption_description)
                        
                        if texts_to_analyze:
                            fake_news_result = fake_news_engine.analyze_multiple_sources(texts_to_analyze)
                        else:
                            st.warning("⚠️ No text provided for fake news analysis. Please enter text above.")

                    # NEW: Microsoft Azure Content Safety (Independent) - analyze combined text
                    content_safety_result = None
                    combined_text = None
                    if news_headline or caption_description:
                        combined_text = " ".join(filter(None, [news_headline, caption_description]))
                        try:
                            cs_engine = ContentSafetyEngine()
                            content_safety_result = cs_engine.analyze(combined_text)
                        except Exception:
                            content_safety_result = None
                    
                    # Module 4: Blockchain Registration (if requested)
                    blockchain_record = None
                    if register_blockchain and generate_protection:
                        blockchain = BlockchainTraceability()
                        blockchain_record = blockchain.register_fingerprint(
                            fingerprint_data,
                            protection_flag=generate_protection,
                            consent_status=True
                        )
                    
                    # NEW: Trust & Risk Aggregator (Combines all module results)
                    trust_aggregation = None
                    if analysis_result or fake_news_result or content_safety_result:
                        aggregator = TrustAggregator()
                        trust_aggregation = aggregator.aggregate(
                            image_analysis=analysis_result,
                            fake_news_analysis=fake_news_result,
                            sensitive_content=content_safety_result if content_safety_result else (analysis_result if analysis_result and analysis_result.get('classification') == 'sensitive' else None)
                        )
                    
                    # Store results in session state
                    st.session_state.analysis_result = analysis_result
                    st.session_state.fake_news_result = fake_news_result
                    st.session_state.content_safety_result = content_safety_result
                    st.session_state.trust_aggregation = trust_aggregation
                    st.session_state.fingerprint_data = fingerprint_data
                    st.session_state.blockchain_record = blockchain_record
                    
                    # Add to protected fingerprints if protection requested
                    if generate_protection:
                        st.session_state.protected_fingerprints.append({
                            'fingerprint': fingerprint_data,
                            'timestamp': blockchain_record['timestamp'] if blockchain_record else None,
                            'protection_flag': True
                        })
                    
                    # Display Multi-Modal Results
                    display_multi_modal_results(
                        analysis_result,
                        fake_news_result,
                        trust_aggregation,
                        fingerprint_data,
                        blockchain_record,
                        generate_protection
                    )
                    
                    # Cleanup - ensure file is deleted (if image was uploaded)
                    if upload_handler and media_data:
                        upload_handler.cleanup(media_data)
                    
                    st.success("✅ Multi-modal analysis complete!" + (" Media processed securely and removed from memory." if uploaded_file else ""))
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.code(traceback.format_exc())
                    st.info("💡 Please try uploading again. Your file was not stored.")
    
    else:
        st.info("👆 **Get Started:** Enter text above for fake news analysis, or upload an image for deepfake detection, or both!")
        st.markdown("""
        ### 📝 Quick Start Guide:
        
        1. **Text-Only Analysis**: Enter a headline or caption above and click "Multi-Modal Analysis"
        2. **Image-Only Analysis**: Upload an image to check for deepfakes
        3. **Combined Analysis**: Upload an image AND enter text for comprehensive threat assessment
        """)


def render_verification_tab():
    """Render the verification tab for re-upload checks"""
    st.header("🔍 Verify Content")
    st.markdown("Check if uploaded content matches protected media in our system.")
    
    uploaded_file = st.file_uploader(
        "Upload content to verify",
        type=['jpg', 'jpeg', 'png', 'webp'],
        key="verify_upload"
    )
    
    if uploaded_file is not None:
        st.image(uploaded_file, use_container_width=True, width=300)
        
        if st.button("🔍 Verify Against Protected Database", type="primary"):
            with st.spinner("Verifying content..."):
                try:
                    # Module 1: Process upload
                    upload_handler = MediaUploadHandler()
                    media_data = upload_handler.process_upload(uploaded_file)
                    
                    # Module 3: Generate fingerprint
                    fingerprint_gen = FingerprintGenerator()
                    fingerprint_data = fingerprint_gen.generate(media_data)
                    
                    # Module 5: Verification
                    verification_engine = VerificationEngine()
                    verification_result = verification_engine.verify(
                        fingerprint_data,
                        st.session_state.protected_fingerprints
                    )
                    
                    # Display verification result
                    if verification_result['match_found']:
                        st.error("⚠️ **PROTECTED CONTENT DETECTED**")
                        st.markdown(f"""
                        <div class="result-box danger-box">
                            <h4>⚠️ This content matches protected media</h4>
                            <p><strong>Match Type:</strong> {verification_result['match_type']}</p>
                            <p><strong>Similarity:</strong> {verification_result['similarity']:.2f}%</p>
                            <p><strong>Status:</strong> Non-consensual sensitive media detected</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success("✅ **NEW & SAFE**")
                        st.markdown(f"""
                        <div class="result-box safe-box">
                            <h4>✅ Content verified as new and safe</h4>
                            <p>This content does not match any protected media in our system.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Cleanup
                    upload_handler.cleanup(media_data)
                    
                except Exception as e:
                    st.error(f"❌ Verification error: {str(e)}")
    
    else:
        st.info("👆 Upload content to verify against protected database.")


def render_dashboard_tab():
    """Render the transparency dashboard"""
    st.header("📊 Transparency Dashboard")
    
    dashboard = ResultDashboard()
    
    # Show last analysis if available
    if st.session_state.trust_aggregation or st.session_state.analysis_result:
        st.subheader("📋 Last Multi-Modal Analysis Result")
        display_multi_modal_results(
            st.session_state.analysis_result,
            st.session_state.fake_news_result,
            st.session_state.trust_aggregation,
            st.session_state.fingerprint_data,
            st.session_state.blockchain_record,
            False
        )
    
    # Statistics
    st.subheader("📈 System Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Protected Media", len(st.session_state.protected_fingerprints))
    
    with col2:
        st.metric("Blockchain Records", 
                1 if st.session_state.blockchain_record else 0)
    
    with col3:
        st.metric("Analyses Performed", 
                1 if st.session_state.analysis_result else 0)
    
    # Protected fingerprints list
    if st.session_state.protected_fingerprints:
        st.subheader("🛡️ Protected Fingerprints")
        for idx, fp_data in enumerate(st.session_state.protected_fingerprints, 1):
            with st.expander(f"Fingerprint #{idx} - {fp_data.get('timestamp', 'N/A')}"):
                st.json({
                    'sha256': fp_data['fingerprint']['sha256'][:32] + "...",
                    'perceptual_hash': fp_data['fingerprint']['perceptual_hash'][:16] + "...",
                    'protection_flag': fp_data.get('protection_flag', False)
                })
    
    # Privacy notice
    st.markdown("---")
    st.markdown("### 🔒 Privacy & Data Handling")
    st.info("""
    **What we store:**
    - Cryptographic fingerprints (hashes only)
    - Timestamps and protection flags
    - Blockchain transaction IDs
    
    **What we do NOT store:**
    - ❌ Original images or videos
    - ❌ Face data or biometric information
    - ❌ Personal identification
    - ❌ Location data
    
    **Data Lifecycle:**
    - Media files are processed in memory only
    - Files are automatically deleted after processing
    - Only fingerprints are retained for protection matching
    """)


if __name__ == "__main__":
    main()
