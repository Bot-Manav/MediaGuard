"""
AI Analysis Engine (Azure-only)

This module provides `AIAnalysisEngine` which performs media analysis
using Microsoft Foundry / Azure AI Content Safety (and optionally
Language) endpoints configured via environment variables.

It intentionally does NOT use any local ML models, joblib, or cv2.
Allowed dependencies: requests, numpy, PIL, os
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import io
import requests
import json
from PIL import Image


class AIAnalysisEngine:
    """AI analysis engine using Azure Content Safety APIs only.

    Environment variables used:
    - AZURE_LANGUAGE_KEY
    - AZURE_LANGUAGE_ENDPOINT
    - AZURE_CONTENT_SAFETY_KEY
    - AZURE_CONTENT_SAFETY_ENDPOINT
    """

    def __init__(self) -> None:
        # Read configuration from environment
        self.language_key = os.getenv('AZURE_LANGUAGE_KEY')
        self.language_endpoint = os.getenv('AZURE_LANGUAGE_ENDPOINT')
        self.cs_key = os.getenv('AZURE_CONTENT_SAFETY_KEY')
        self.cs_endpoint = os.getenv('AZURE_CONTENT_SAFETY_ENDPOINT')

        # Startup logging for endpoints (helps verify configuration)
        print("Azure Language Endpoint:", self.language_endpoint)
        print("Azure Content Safety Endpoint:", self.cs_endpoint)

        # Basic validation - we rely entirely on remote APIs
        if not self.cs_key or not self.cs_endpoint:
            print("Warning: Azure Content Safety key/endpoint not fully configured.")

    def analyze(self, media_data: Dict[str, Any], check_deepfake: bool = True, check_sensitive: bool = True) -> Dict[str, Any]:
        """Analyze an image using Azure Content Safety.

        Args:
            media_data: dict containing at least 'image' with a PIL.Image.Image
            check_deepfake: whether to request manipulation/deepfake indicators
            check_sensitive: whether to request sensitive content classification

        Returns: standard result dict with keys 'classification', 'confidence', 'details', 'recommendations'
        """
        image: Optional[Image.Image] = media_data.get('image')

        if image is None:
            return {
                'classification': 'unknown',
                'confidence': 0.0,
                'details': {'error': 'No image data available'},
                'recommendations': []
            }

        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Prepare payload (JPEG bytes)
        img_bytes = self._image_to_jpeg_bytes(image)

        # Default results
        manipulation_score = 0.0
        sensitive_score = 0.0
        manipulation_indicators: List[str] = []
        sensitive_indicators: List[str] = []
        raw_foundry: Optional[Dict[str, Any]] = None

        # Call Content Safety API once and extract required pieces
        try:
            if not self.cs_endpoint or not self.cs_key:
                raise RuntimeError('Azure Content Safety endpoint/key not configured')

            resp_json = self._post_image_to_content_safety(img_bytes)
            raw_foundry = resp_json

            # Parse the response to find manipulation & sensitive indicators/scores
            manipulation_score, manipulation_indicators, sensitive_score, sensitive_indicators = (
                self._parse_content_safety_response(resp_json)
            )

        except Exception as e:
            # On any Foundry failure we must not fallback to local models
            details = {
                'foundry_error': str(e),
                'foundry_response': raw_foundry
            }
            return {
                'classification': 'unknown',
                'confidence': 0.0,
                'details': details,
                'recommendations': []
            }

        # Decide final classification and confidence
        classification, confidence = self._classify(manipulation_score, sensitive_score)

        details = {
            'deepfake_score': manipulation_score,
            'sensitive_score': sensitive_score,
            'indicators': {
                'manipulation': manipulation_indicators,
                'sensitive': sensitive_indicators
            },
            'foundry_response': raw_foundry
        }

        recommendations = self._generate_recommendations(classification)

        return {
            'classification': classification,
            'confidence': confidence,
            'details': details,
            'recommendations': recommendations
        }

    def _image_to_jpeg_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL image to JPEG bytes for upload."""
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=90)
        return buf.getvalue()

    def _post_image_to_content_safety(self, img_bytes: bytes) -> Dict[str, Any]:
        """POST image bytes to the configured Content Safety endpoint.

        The method assumes that `self.cs_endpoint` is a fully-qualified URL
        for the Content Safety image analysis route. Headers use the
        subscription key configured in `AZURE_CONTENT_SAFETY_KEY`.
        """
        headers = {
            'Ocp-Apim-Subscription-Key': self.cs_key,
            'Content-Type': 'application/octet-stream'
        }

        # Send request
        resp = requests.post(self.cs_endpoint, headers=headers, data=img_bytes, timeout=30)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            # Attempt to include response text for debugging
            raise RuntimeError(f'Content Safety API HTTP error: {e} - {resp.text}')

        # Parse JSON if possible
        try:
            return resp.json()
        except Exception:
            # If not JSON, return text wrapper
            return {'text': resp.text}

    def _parse_content_safety_response(self, resp_json: Dict[str, Any]) -> Tuple[float, List[str], float, List[str]]:
        """Extract manipulation and sensitive scores + human-readable indicators.

        This function attempts to be robust to varying shapes of Foundry
        responses by scanning the JSON for keys or numeric scores that
        indicate manipulation/deepfake or sensitive content.
        Returns: (manipulation_score, manipulation_indicators, sensitive_score, sensitive_indicators)
        """
        manipulation_score = 0.0
        sensitive_score = 0.0
        manipulation_indicators: List[str] = []
        sensitive_indicators: List[str] = []

        def walk(obj: Any, path: str = '') -> None:
            nonlocal manipulation_score, sensitive_score
            if isinstance(obj, dict):
                for k, v in obj.items():
                    key_lower = k.lower() if isinstance(k, str) else ''
                    # Heuristics: keys mentioning manipulation/deepfake
                    if isinstance(v, (int, float)):
                        val = float(v)
                        if 'manipul' in key_lower or 'deepfake' in key_lower or 'synthetic' in key_lower:
                            manipulation_score = max(manipulation_score, min(val * 100 if val <= 1.0 else val, 100.0))
                            manipulation_indicators.append(f"{k}: {val}")
                        if any(x in key_lower for x in ['sensitive', 'adult', 'nudity', 'sexual', 'racy', 'privacy', 'violence']):
                            sensitive_score = max(sensitive_score, min(val * 100 if val <= 1.0 else val, 100.0))
                            sensitive_indicators.append(f"{k}: {val}")
                    elif isinstance(v, str):
                        # textual signals
                        if any(tok in v.lower() for tok in ['manipulat', 'deepfake', 'synthetic']):
                            manipulation_indicators.append(f"{k}: {v}")
                        if any(tok in v.lower() for tok in ['sensitive', 'nudity', 'sexual', 'private', 'privacy']):
                            sensitive_indicators.append(f"{k}: {v}")
                    else:
                        walk(v, path + '/' + k)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    walk(item, path + f'[{i}]')

        walk(resp_json)

        # Additional: look into top-level categories if present
        categories = resp_json.get('categories') if isinstance(resp_json, dict) else None
        if isinstance(categories, dict):
            for cat, meta in categories.items():
                cat_lower = cat.lower()
                if any(tok in cat_lower for tok in ['deepfake', 'manipulation', 'synthetic']):
                    # meta may contain 'score' or 'confidence'
                    score = self._safe_extract_score(meta)
                    if score is not None:
                        manipulation_score = max(manipulation_score, score)
                        manipulation_indicators.append(f"category:{cat}={score}")
                if any(tok in cat_lower for tok in ['sexual', 'nudity', 'sensitive', 'privacy', 'violence', 'adult', 'racy']):
                    score = self._safe_extract_score(meta)
                    if score is not None:
                        sensitive_score = max(sensitive_score, score)
                        sensitive_indicators.append(f"category:{cat}={score}")

        # Normalize to 0-100 and ensure floats
        manipulation_score = float(min(max(manipulation_score, 0.0), 100.0))
        sensitive_score = float(min(max(sensitive_score, 0.0), 100.0))

        # If no textual indicators were found, add a summary key if present
        if not manipulation_indicators and isinstance(resp_json, dict):
            if 'manipulation' in resp_json:
                manipulation_indicators.append(str(resp_json.get('manipulation')))
        if not sensitive_indicators and isinstance(resp_json, dict):
            if 'sensitive' in resp_json:
                sensitive_indicators.append(str(resp_json.get('sensitive')))

        return manipulation_score, manipulation_indicators, sensitive_score, sensitive_indicators

    def _safe_extract_score(self, meta: Any) -> Optional[float]:
        """Try to extract a numeric score from various possible meta shapes."""
        if meta is None:
            return None
        if isinstance(meta, (int, float)):
            val = float(meta)
            return val if val > 1.0 else val * 100.0
        if isinstance(meta, dict):
            for k in ['score', 'confidence', 'severity']:
                v = meta.get(k)
                if isinstance(v, (int, float)):
                    return float(v) if v > 1.0 else float(v) * 100.0
        return None

    def _classify(self, manipulation_score: float, sensitive_score: float) -> Tuple[str, float]:
        """Simple decision logic to convert scores into a class and confidence."""
        # Priority: deepfake/manipulation > sensitive
        if manipulation_score >= 50.0:
            classification = 'deepfake'
            confidence = min(manipulation_score, 98.5)
        elif sensitive_score >= 40.0:
            classification = 'sensitive'
            confidence = min(sensitive_score, 98.0)
        else:
            classification = 'real_safe'
            raw_confidence = 100.0 - max(manipulation_score, sensitive_score)
            confidence = min(raw_confidence, 98.0)
        return classification, float(confidence)

    def _generate_recommendations(self, classification: str) -> List[str]:
        """Return human-readable recommendations for each classification."""
        if classification == 'deepfake':
            return [
                '⚠️ This content shows signs of manipulation',
                '🔍 Verify source and provenance before sharing',
                '🛡️ Consider reporting and avoid redistribution'
            ]
        if classification == 'sensitive':
            return [
                '🔒 This content may contain sensitive or private material',
                '🛡️ Consider generating a protection fingerprint',
                '⚠️ Exercise caution when sharing'
            ]
        return ['✅ Content appears authentic', '📝 No immediate concerns detected']
