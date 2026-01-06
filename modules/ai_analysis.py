"""
Module: AI Analysis Engine (Microsoft AI-only)
Classifies uploaded media into:
- 🟢 Real & Safe
- 🔴 Deepfake / Manipulated
- 🟡 Real but Sensitive (Private)
Uses Microsoft Foundry / Azure AI exclusively. No local ML models are referenced.
"""

import os
import io
import requests
from typing import Dict, Any
from PIL import Image


class AIAnalysisEngine:
    """Microsoft AI-powered analysis engine for media classification"""

    def __init__(self):
        # Azure / Foundry credentials from environment
        self.azure_language_key = os.getenv("AZURE_LANGUAGE_KEY")
        self.azure_language_endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")
        self.azure_content_safety_key = os.getenv("AZURE_CONTENT_SAFETY_KEY")
        self.azure_content_safety_endpoint = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")

        print("Azure Language Endpoint:", self.azure_language_endpoint)
        print("Azure Content Safety Endpoint:", self.azure_content_safety_endpoint)

    def analyze(self, media_data: Dict[str, Any],
                check_deepfake: bool = True,
                check_sensitive: bool = True) -> Dict[str, Any]:
        """
        Analyze media and return classification results using Microsoft AI.
        """
        image: Image.Image = media_data.get('image')
        if image is None:
            return {
                'classification': 'unknown',
                'confidence': 0.0,
                'details': {'error': 'No image data available'},
                'recommendations': []
            }

        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert image to bytes for API
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()

        results = {
            'deepfake_score': 0.0,
            'sensitive_score': 0.0,
            'manipulation_indicators': [],
            'sensitive_indicators': [],
            'foundry_used': False,
            'foundry_error': None
        }

        # ---- Deepfake / Manipulation Analysis ----
        if check_deepfake:
            deepfake_result = self._call_content_safety_api(img_bytes, analysis_type='deepfake')
            if 'error' in deepfake_result:
                results['foundry_error'] = f"Deepfake API error: {deepfake_result['error']}"
            else:
                results['deepfake_score'] = deepfake_result.get('score', 0.0)
                results['manipulation_indicators'] = deepfake_result.get('indicators', [])
                results['foundry_used'] = True

        # ---- Sensitive Content Analysis ----
        if check_sensitive:
            sensitive_result = self._call_content_safety_api(img_bytes, analysis_type='sensitive')
            if 'error' in sensitive_result:
                results['foundry_error'] = (results['foundry_error'] or '') + f" Sensitive API error: {sensitive_result['error']}"
            else:
                results['sensitive_score'] = sensitive_result.get('score', 0.0)
                results['sensitive_indicators'] = sensitive_result.get('indicators', [])
                results['foundry_used'] = True

        # ---- Classification Logic ----
        classification, confidence, details_text = self._classify_results(results)

        # ---- Generate Recommendations ----
        recommendations = self._generate_recommendations(classification)

        return {
            'classification': classification,
            'confidence': confidence,
            'details': {
                'deepfake_score': results['deepfake_score'],
                'sensitive_score': results['sensitive_score'],
                'indicators': {
                    'manipulation': results['manipulation_indicators'],
                    'sensitive': results['sensitive_indicators']
                },
                'foundry_used': results['foundry_used'],
                'foundry_error': results['foundry_error']
            },
            'recommendations': recommendations
        }

    def _call_content_safety_api(self, img_bytes: bytes, analysis_type: str) -> Dict[str, Any]:
        """Call Microsoft Content Safety API for deepfake or sensitive analysis"""
        if not self.azure_content_safety_key or not self.azure_content_safety_endpoint:
            return {'error': 'Azure Content Safety Key/Endpoint not configured'}

        headers = {
            "Ocp-Apim-Subscription-Key": self.azure_content_safety_key,
            "Content-Type": "application/octet-stream"
        }

        try:
            response = requests.post(
                self.azure_content_safety_endpoint,
                headers=headers,
                data=img_bytes,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            # Extract scores & indicators depending on analysis_type
            if analysis_type == 'deepfake':
                return {
                    'score': data.get('deepfake_score', 0.0),
                    'indicators': data.get('manipulation_indicators', [])
                }
            elif analysis_type == 'sensitive':
                return {
                    'score': data.get('sensitive_score', 0.0),
                    'indicators': data.get('sensitive_indicators', [])
                }
            else:
                return {'error': 'Unknown analysis_type'}

        except Exception as e:
            return {'error': str(e)}

    def _classify_results(self, results: Dict[str, Any]) -> tuple:
        """Classify media based on Microsoft AI scores"""
        deepfake_score = results['deepfake_score']
        sensitive_score = results['sensitive_score']

        if results['foundry_error']:
            return "unknown", 0.0, f"Foundry API error: {results['foundry_error']}"

        if deepfake_score > 50:
            classification = 'deepfake'
            confidence = min(deepfake_score, 98.0)
            details = "High likelihood of manipulation or deepfake"
        elif sensitive_score > 40:
            classification = 'sensitive'
            confidence = min(sensitive_score, 98.0)
            details = "Content appears to be sensitive/private"
        else:
            classification = 'real_safe'
            confidence = 98.0 - max(deepfake_score, sensitive_score)
            details = "Content appears authentic and safe"

        return classification, confidence, details

    def _generate_recommendations(self, classification: str) -> list:
        """Generate human-readable recommendations based on classification"""
        recommendations = []

        if classification == 'deepfake':
            recommendations.extend([
                "⚠️ This content shows signs of manipulation",
                "🔍 Verify source before sharing",
                "🛡️ Consider reporting if used maliciously"
            ])
        elif classification == 'sensitive':
            recommendations.extend([
                "🔒 This content may be sensitive",
                "🛡️ Consider generating protection fingerprint",
                "⚠️ Be cautious about sharing"
            ])
        else:
            recommendations.extend([
                "✅ Content appears authentic",
                "📝 No immediate concerns detected"
            ])
        return recommendations
