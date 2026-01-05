"""
Module: Trust & Risk Aggregator
The brain of the system - combines results from all independent modules.

This module:
- Does NOT analyze raw data
- Only combines results from modules
- Generates overall trust score and threat category
"""

from typing import Dict, Any, Optional, List


class TrustAggregator:
    """
    Aggregates results from multiple analysis modules to generate
    a comprehensive trust and risk assessment.
    """
    
    def __init__(self):
        # Weight configuration for different modules
        self.weights = {
            'image_analysis': 0.4,  # 40% weight
            'fake_news': 0.4,        # 40% weight
            'sensitive_content': 0.2  # 20% weight
        }
    
    def aggregate(self,
                 image_analysis: Optional[Dict[str, Any]] = None,
                 fake_news_analysis: Optional[Dict[str, Any]] = None,
                 sensitive_content: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Aggregate results from all modules into a final trust assessment.
        
        Args:
            image_analysis: Results from image analysis module
            fake_news_analysis: Results from fake news detection module
            sensitive_content: Results from sensitive content detection
            
        Returns:
            {
                'overall_trust_risk': float (0-100),
                'threat_category': str,
                'confidence': float (0-100),
                'module_results': {
                    'image': {...},
                    'fake_news': {...},
                    'sensitive': {...}
                },
                'explanation': str,
                'recommendations': List[str]
            }
        """
        module_results = {}
        risk_scores = []
        explanations = []
        
        # Process image analysis
        image_risk = 0.0
        if image_analysis:
            classification = image_analysis.get('classification', 'unknown')
            confidence = image_analysis.get('confidence', 0.0)
            
            if classification == 'deepfake':
                image_risk = 80.0  # High risk for deepfakes
                explanations.append(f"Image manipulation detected ({confidence:.1f}% confidence)")
            elif classification == 'sensitive':
                image_risk = 40.0  # Moderate risk
                explanations.append(f"Sensitive content detected ({confidence:.1f}% confidence)")
            elif classification == 'real_safe':
                image_risk = 20.0  # Low risk
                explanations.append(f"Image appears authentic ({confidence:.1f}% confidence)")
            else:
                image_risk = 50.0  # Unknown/uncertain
                explanations.append("Image analysis inconclusive")
            
            module_results['image'] = {
                'risk': image_risk,
                'classification': classification,
                'confidence': confidence
            }
            risk_scores.append(('image', image_risk, self.weights['image_analysis']))
        
        # Process fake news analysis
        fake_news_risk = 0.0
        if fake_news_analysis:
            fake_news_likelihood = fake_news_analysis.get('fake_news_likelihood', 0.0)
            credibility_score = fake_news_analysis.get('credibility_score', 50.0)
            
            # Fake news likelihood directly contributes to risk
            fake_news_risk = fake_news_likelihood
            
            if fake_news_likelihood > 50:
                prefix = "High" if fake_news_likelihood > 75 else "Moderate"
                explanations.append(f"{prefix} fake news likelihood ({fake_news_likelihood:.1f}%)")
            elif credibility_score > 60:
                explanations.append(f"Content appears credible ({credibility_score:.1f}%)")
            else:
                explanations.append("Text analysis results are uncertain")
            
            module_results['fake_news'] = {
                'risk': fake_news_risk,
                'fake_news_likelihood': fake_news_likelihood,
                'credibility_score': credibility_score
            }
            risk_scores.append(('fake_news', fake_news_risk, self.weights['fake_news']))
        
        # Process sensitive content
        sensitive_risk = 0.0
        if sensitive_content:
            sensitive_score = sensitive_content.get('sensitive_score', 0.0)
            sensitive_risk = sensitive_score
            
            if sensitive_score > 50:
                explanations.append(f"Sensitive content detected ({sensitive_score:.1f}%)")
            
            module_results['sensitive'] = {
                'risk': sensitive_risk,
                'score': sensitive_score
            }
            risk_scores.append(('sensitive', sensitive_risk, self.weights['sensitive_content']))
        
        # Calculate weighted overall trust risk
        if risk_scores:
            weighted_sum = sum(risk * weight for _, risk, weight in risk_scores)
            total_weight = sum(weight for _, _, weight in risk_scores)
            overall_trust_risk = weighted_sum / total_weight if total_weight > 0 else 50.0
        else:
            overall_trust_risk = 50.0  # Default to medium risk if no data
        
        # Determine threat category
        threat_category = self._determine_threat_category(
            overall_trust_risk,
            image_analysis,
            fake_news_analysis
        )
        
        # Calculate overall confidence
        confidences = []
        if image_analysis:
            confidences.append(image_analysis.get('confidence', 0))
        if fake_news_analysis:
            # Use inverse of uncertainty as confidence
            uncertain = fake_news_analysis.get('uncertain', 50)
            confidences.append(100 - uncertain)
        
        overall_confidence = sum(confidences) / len(confidences) if confidences else 50.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_trust_risk,
            threat_category,
            image_analysis,
            fake_news_analysis
        )
        
        return {
            'overall_trust_risk': round(overall_trust_risk, 1),
            'threat_category': threat_category,
            'confidence': round(overall_confidence, 1),
            'module_results': module_results,
            'explanation': ' | '.join(explanations) if explanations else 'No analysis data available',
            'recommendations': recommendations
        }
    
    def _determine_threat_category(self,
                                   overall_risk: float,
                                   image_analysis: Optional[Dict[str, Any]],
                                   fake_news_analysis: Optional[Dict[str, Any]]) -> str:
        """Determine the primary threat category based on aggregated results"""
        
        if overall_risk >= 65:
            # High risk scenarios
            if image_analysis and image_analysis.get('classification') == 'deepfake':
                if fake_news_analysis and fake_news_analysis.get('fake_news_likelihood', 0) > 50:
                    return "Severe: Manipulated Media + Misinformation"
                else:
                    return "High: Manipulated Media"
            elif fake_news_analysis and fake_news_analysis.get('fake_news_likelihood', 0) > 50:
                return "High: Misinformation"
            else:
                return "High: Multiple Risk Factors"
        
        elif overall_risk >= 40:
            # Medium risk scenarios
            if image_analysis and image_analysis.get('classification') == 'deepfake':
                return "Medium: Potential Media Manipulation"
            elif fake_news_analysis and fake_news_analysis.get('fake_news_likelihood', 0) > 40:
                return "Medium: Potential Misinformation"
            else:
                return "Medium: Moderate Risk"
        
        elif overall_risk >= 30:
            # Low-medium risk
            return "Low-Medium: Some Concerns"
        
        else:
            # Low risk
            return "Low: Generally Trustworthy"
    
    def _generate_recommendations(self,
                                 overall_risk: float,
                                 threat_category: str,
                                 image_analysis: Optional[Dict[str, Any]],
                                 fake_news_analysis: Optional[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if overall_risk >= 70:
            recommendations.append("⚠️ **High Risk**: Exercise extreme caution with this content")
            recommendations.append("🔍 Verify information through multiple independent sources")
            recommendations.append("🚫 Do not share without verification")
        
        elif overall_risk >= 50:
            recommendations.append("⚠️ **Moderate Risk**: Verify before sharing")
            recommendations.append("📚 Check fact-checking websites")
        
        if image_analysis and image_analysis.get('classification') == 'deepfake':
            recommendations.append("🖼️ Image shows signs of manipulation - verify source")
        
        if fake_news_analysis and fake_news_analysis.get('fake_news_likelihood', 0) > 60:
            recommendations.append("📰 Text contains indicators of misinformation")
            recommendations.append("✅ Cross-reference with credible news sources")
        
        if overall_risk < 30:
            recommendations.append("✅ Content appears generally trustworthy")
            recommendations.append("📝 Still recommended to verify critical claims")
        
        if not recommendations:
            recommendations.append("ℹ️ Review analysis results for specific concerns")
        
        return recommendations

