"""
Module 7: Result & Transparency Dashboard
Displays analysis results, confidence scores, and protection status.
"""

import streamlit as st
from typing import Dict, Any, Optional


class ResultDashboard:
    """Displays analysis results and transparency information"""
    
    def display_results(self,
                       analysis_result: Dict[str, Any],
                       fingerprint_data: Dict[str, Any],
                       blockchain_record: Optional[Dict[str, Any]],
                       protection_enabled: bool):
        """Display comprehensive analysis results"""
        
        # Classification result
        classification = analysis_result.get('classification', 'unknown')
        confidence = analysis_result.get('confidence', 0.0)
        
        # Display classification with appropriate styling
        st.markdown("## 📊 Analysis Results")
        
        if classification == 'real_safe':
            self._display_safe_result(confidence, analysis_result)
        elif classification == 'deepfake':
            self._display_deepfake_result(confidence, analysis_result)
        elif classification == 'sensitive':
            self._display_sensitive_result(confidence, analysis_result)
        else:
            st.warning("⚠️ Classification: Unknown")
        
        # Fingerprint information
        st.markdown("---")
        st.markdown("## 🔐 Fingerprint Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cryptographic Hash")
            st.code(fingerprint_data.get('sha256', 'N/A')[:64] + "...", language=None)
            st.caption("SHA-256 hash of the media file")
        
        with col2:
            st.subheader("Perceptual Hash")
            st.code(fingerprint_data.get('perceptual_hash', 'N/A'), language=None)
            st.caption("Hash for detecting similar images")
        
        # Face embedding (if available)
        face_hash = fingerprint_data.get('face_embedding_hash')
        if face_hash:
            st.subheader("Face Embedding Hash")
            st.code(face_hash[:32] + "...", language=None)
            st.caption("Numerical representation of detected faces")
        
        # Blockchain record
        if blockchain_record:
            st.markdown("---")
            st.markdown("## ⛓️ Blockchain Record")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Block Number", blockchain_record.get('block_number', 'N/A'))
            
            with col2:
                st.metric("Protection Flag", "✅ Enabled" if blockchain_record.get('protection_flag') else "❌ Disabled")
            
            with col3:
                st.metric("Consent Status", "✅ Granted" if blockchain_record.get('consent_status') else "❌ Not Granted")
            
            with st.expander("📋 Blockchain Details"):
                st.json({
                    'transaction_id': blockchain_record.get('transaction_id', 'N/A'),
                    'timestamp': blockchain_record.get('timestamp', 'N/A'),
                    'blockchain_type': blockchain_record.get('blockchain_type', 'N/A'),
                    'block_hash': blockchain_record.get('block_hash', 'N/A')[:32] + "..."
                })
        
        # Protection status
        if protection_enabled:
            st.markdown("---")
            st.success("🛡️ **Protection Fingerprint Generated Successfully**")
            st.info("""
            Your media fingerprint has been registered and protected.
            - Fingerprint stored securely
            - Re-upload detection enabled
            - Blockchain record created (if enabled)
            """)
        
        # Recommendations
        recommendations = analysis_result.get('recommendations', [])
        if recommendations:
            st.markdown("---")
            st.markdown("## 💡 Recommendations")
            for rec in recommendations:
                st.markdown(f"- {rec}")
        
        # Transparency information
        st.markdown("---")
        st.markdown("## 🔒 Privacy & Transparency")
        st.info("""
        **What was processed:**
        - Media file analyzed in memory
        - Fingerprints generated (hashes only)
        - No raw image data stored
        
        **What was stored:**
        - Cryptographic fingerprints (hashes)
        - Protection flags
        - Timestamps
        
        **What was NOT stored:**
        - ❌ Original images
        - ❌ Face data
        - ❌ Personal information
        """)
    
    def _display_safe_result(self, confidence: float, analysis_result: Dict[str, Any]):
        """Display result for safe/real content"""
        st.markdown("""
        <div class="result-box safe-box">
            <h3>🟢 Real & Safe</h3>
            <p><strong>Confidence:</strong> {:.1f}%</p>
            <p>Content appears to be authentic with no significant manipulation detected.</p>
        </div>
        """.format(confidence), unsafe_allow_html=True)
    
    def _display_deepfake_result(self, confidence: float, analysis_result: Dict[str, Any]):
        """Display result for deepfake/manipulated content"""
        st.markdown("""
        <div class="result-box danger-box">
            <h3>🔴 Deepfake / Manipulated</h3>
            <p><strong>Confidence:</strong> {:.1f}%</p>
            <p>⚠️ This content shows signs of manipulation or deepfake generation.</p>
        </div>
        """.format(confidence), unsafe_allow_html=True)
        
        # Show indicators if available
        indicators = analysis_result.get('details', {}).get('indicators', {}).get('manipulation', [])
        if indicators:
            st.warning("**Manipulation Indicators Detected:**")
            for indicator in indicators:
                st.markdown(f"- {indicator}")
    
    def _display_sensitive_result(self, confidence: float, analysis_result: Dict[str, Any]):
        """Display result for sensitive content"""
        st.markdown("""
        <div class="result-box warning-box">
            <h3>🟡 Real but Sensitive</h3>
            <p><strong>Confidence:</strong> {:.1f}%</p>
            <p>Content appears authentic but may contain sensitive/private information.</p>
        </div>
        """.format(confidence), unsafe_allow_html=True)
        
        st.info("💡 **Recommendation:** Consider generating a protection fingerprint to prevent unauthorized re-uploads.")

