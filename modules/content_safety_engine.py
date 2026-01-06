import os
from typing import Dict, Any

from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.core.credentials import AzureKeyCredential


class ContentSafetyEngine:
    """
    Microsoft Azure Content Safety integration
    Free-tier compatible
    """

    def __init__(self):
        endpoint = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
        key = os.getenv("AZURE_CONTENT_SAFETY_KEY")

        if not endpoint or not key:
            raise RuntimeError("Azure Content Safety credentials not configured")

        self.client = ContentSafetyClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for harmful / risky content
        """
        if not text.strip():
            return None

        request = AnalyzeTextOptions(text=text)
        response = self.client.analyze_text(request)

        categories = {}
        max_severity = 0

        for c in response.categories_analysis:
            categories[c.category] = {
                "severity": c.severity,
                "confidence": c.confidence
            }
            max_severity = max(max_severity, c.severity)

        return {
            "source": "Azure Content Safety",
            "max_severity": max_severity,
            "categories": categories,
            "risk_score": min((max_severity / 7) * 100, 100)
        }
