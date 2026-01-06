import os
from typing import Optional, Dict, Any


class ContentSafetyEngine:
    """Wrapper around Azure Content Safety SDK.

    This class attempts to use the `azure-ai-contentsafety` SDK when available
    and falls back to a safe, non-failing default when the SDK or credentials
    are missing. The goal is to be non-destructive and not to crash the app.
    """

    def __init__(self):
        self.endpoint = os.getenv('AZURE_CONTENT_SAFETY_ENDPOINT')
        self.key = os.getenv('AZURE_CONTENT_SAFETY_KEY')
        self.client = None
        try:
            # Import lazily so the module can be imported without the SDK installed
            from azure.ai.contentsafety import ContentSafetyClient
            from azure.core.credentials import AzureKeyCredential

            if self.endpoint and self.key:
                try:
                    self.client = ContentSafetyClient(self.endpoint, AzureKeyCredential(self.key))
                except Exception:
                    # If client creation fails (invalid creds, network, etc.) keep client None
                    self.client = None
        except Exception:
            # SDK not installed or import failed - keep client None
            self.client = None

    def analyze(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyze text with Azure Content Safety and return a normalized dict.

        Returns None for empty input. Always returns a dict with the following
        shape when analysis is available or when returning a safe default:

        {
            "source": "Azure Content Safety",
            "max_severity": int (0-7),
            "risk_score": float (0-100),
            "sensitive_score": float (0-100),
            "categories": { "<category>": {"severity": int, "confidence": float} }
        }
        """
        if not text or not text.strip():
            return None

        # Default (safe) response if SDK or credentials unavailable
        default = {
            "source": "Azure Content Safety",
            "max_severity": 0,
            "risk_score": 0.0,
            "sensitive_score": 0.0,
            "categories": {}
        }

        if not self.client:
            return default

        try:
            # Try common client methods - be permissive about SDK versions
            response = None
            if hasattr(self.client, 'analyze_text'):
                try:
                    from azure.ai.contentsafety.models import TextAnalysisOptions
                    response = self.client.analyze_text(
                    TextAnalysisOptions(text=text)
                )
                except Exception:
                    try:
                        response = self.client.analyze_text(text)
                    except Exception:
                        response = None

            elif hasattr(self.client, 'analyze'):
                response = self.client.analyze(text)

            if response is None:
                return default

            # Normalize response: try to extract labeled categories, severities, confidences
            categories: Dict[str, Dict[str, Any]] = {}
            max_severity = 0
            risk_score = 0.0

            # Many responses expose a `results` iterable
            results = getattr(response, 'results', None) or getattr(response, 'classification', None) or []

            for r in results or []:
                # Best-effort extraction of fields without assuming exact SDK types
                label = getattr(r, 'category', None) or getattr(r, 'label', None) or getattr(r, 'id', None) or str(r)
                severity = getattr(r, 'severity', None) or getattr(r, 'score', None) or 0
                confidence = getattr(r, 'confidence', None) or getattr(r, 'score', None) or 0.0

                try:
                    sev_int = int(severity)
                except Exception:
                    try:
                        sev_int = int(float(severity))
                    except Exception:
                        sev_int = 0

                try:
                    conf_f = float(confidence)
                    if conf_f <= 1:
                        conf_f = conf_f * 100.0
                except Exception:
                    conf_f = 0.0

                categories[str(label)] = {"severity": sev_int, "confidence": round(conf_f, 1)}
                if sev_int > max_severity:
                    max_severity = sev_int

            # Attempt to read an overall score if present
            rs = getattr(response, 'risk_score', None) or getattr(response, 'score', None) or 0.0
            try:
                risk_score = float(rs)
                if risk_score <= 1:
                    risk_score = risk_score * 100.0
            except Exception:
                risk_score = 0.0
            # Fallback: derive risk score from max severity if Azure didn't provide one
            if risk_score == 0.0 and categories:
                risk_score = (max_severity / 7) * 100

            # Map to a single sensitive_score field for aggregator (0-100)
            sensitive_score = risk_score

            return {
                "source": "Azure Content Safety",
                "max_severity": int(max_severity),
                "risk_score": round(float(risk_score), 1),
                "sensitive_score": round(float(sensitive_score), 1),
                "categories": categories
            }

        except Exception:
            # Never raise - return safe default on any unexpected issue
            return default
