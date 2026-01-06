import os
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.core.credentials import AzureKeyCredential


def analyze_text_safety(text: str):
    """
    Analyze text using Azure Content Safety
    Returns risk categories and severity
    """

    endpoint = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
    key = os.getenv("AZURE_CONTENT_SAFETY_KEY")

    if not endpoint or not key:
        return {
            "error": "Azure Content Safety credentials not configured"
        }

    client = ContentSafetyClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    request = AnalyzeTextOptions(text=text)
    response = client.analyze_text(request)

    results = {}

    for category in response.categories_analysis:
        results[category.category] = {
            "severity": category.severity,
            "confidence": category.confidence
        }

    return results
