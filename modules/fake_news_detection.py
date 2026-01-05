"""
Module: Fake News Detection
Independent module for detecting misleading, manipulated, or false textual information.

This module is SEPARATE from image analysis because:
- An image can be real but have fake news
- An image can be fake but have true news
- They must be evaluated independently
"""

import re
import os
import joblib
import numpy as np
from typing import Dict, Any, List, Optional
from collections import Counter


class FakeNewsDetectionEngine:
    """
    Detects fake news and misinformation in textual content.
    
    Evaluates signals such as:
    - Emotional language intensity
    - Sensational keywords
    - Claim verifiability patterns
    - Linguistic manipulation markers
    """
    
    def __init__(self):
        # ML Model paths
        self.model_path = "models/fake_news_model.pkl"
        self.vectorizer_path = "models/fake_news_vectorizer.pkl"
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        
        self._load_models()

    def _load_models(self):
        """Load the trained ML models if they exist."""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.feature_names = self.vectorizer.get_feature_names_out()
                print("ML models for fake news detection loaded successfully.")
        except Exception as e:
            print(f"Error loading ML models: {e}")
    
    def _get_explainability_signals(self, text: str) -> List[str]:
        """Extra feature importance from the model to provide human-readable reasons."""
        if not self.model or not self.vectorizer or not self.feature_names is not None:
            return ["ML explainability unavailable"]
            
        try:
            # Transform text
            tfidf_vec = self.vectorizer.transform([text]).toarray()[0]
            if not np.any(tfidf_vec):
                return ["No recognizable patterns found in text"]
                
            # Multiply TF-IDF values by model coefficients
            # Class 1 is 'Fake', so positive coefficients increase fake likelihood
            importance = tfidf_vec * self.model.coef_[0]
            
            # Get top features
            top_indices = np.argsort(importance)[-5:] # Top 5 words contributing to 'Fake'
            bottom_indices = np.argsort(importance)[:3] # Top words contributing to 'Real'
            
            reasons = []
            for idx in reversed(top_indices):
                if importance[idx] > 0.1: # Significant contribution
                    reasons.append(f"Pattern detected: word '{self.feature_names[idx]}' strongly correlates with misinformation")
                elif importance[idx] > 0:
                    reasons.append(f"Subtle signal: '{self.feature_names[idx]}' contributes to risk profile")
            
            for idx in bottom_indices:
                if importance[idx] < -0.1:
                    reasons.append(f"Trust signal: '{self.feature_names[idx]}' correlates with reliable reporting")
                    
            if not reasons:
                return ["Model identified generic patterns without specific strong indicators"]
            return reasons
        except Exception as e:
            return [f"Explainability error: {str(e)}"]

    def analyze(self, text: str) -> Dict[str, Any]:
        """Pure ML analysis with feature importance explainability."""
        if not text or len(text.strip()) == 0:
            return {
                'fake_news_likelihood': 0.0,
                'credibility_score': 50.0,
                'uncertain': 50.0,
                'explanation': {
                    'signals': ['No text provided'],
                    'keywords_found': [],
                    'emotional_intensity': 0.0,
                    'verifiability_score': 50.0
                }
            }
        
        signals = []
        fake_news_score = 50.0 # Default neutral
        ml_applied = False
        
        if self.model and self.vectorizer:
            try:
                tfidf_text = self.vectorizer.transform([text])
                probs = self.model.predict_proba(tfidf_text)[0]
                fake_news_score = probs[1] * 100
                ml_applied = True
                
                # Get human-readable reasons from ML weights
                signals = self._get_explainability_signals(text)
            except Exception as e:
                signals.append(f"ML Analysis failed: {str(e)}")
        else:
            signals.append("ML Model not loaded - detection currently inactive")
            
        fake_news_likelihood = max(0, min(98.5, fake_news_score))
        credibility_score = 100 - fake_news_likelihood
        
        # Uncertainty: higher when results are near 50%, capped for realism
        uncertain = max(1.5, min(100, 100 - abs(fake_news_likelihood - 50) * 2))
        
        return {
            'fake_news_likelihood': round(fake_news_likelihood, 1),
            'credibility_score': round(credibility_score, 1),
            'uncertain': round(uncertain, 1),
            'explanation': {
                'signals': signals,
                'ml_applied': ml_applied,
                'text_length': len(text.split())
            }
        }
    
    def analyze_multiple_sources(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple text sources and aggregate results.
        """
        if not texts:
            return self.analyze("")
        
        results = [self.analyze(text) for text in texts if text]
        
        if not results:
            return self.analyze("")
        
        total_fake = sum(r['fake_news_likelihood'] for r in results)
        total_cred = sum(r['credibility_score'] for r in results)
        total_uncertain = sum(r['uncertain'] for r in results)
        count = len(results)
        
        all_signals = []
        for r in results:
            all_signals.extend(r['explanation']['signals'])
        
        return {
            'fake_news_likelihood': round(total_fake / count, 1),
            'credibility_score': round(total_cred / count, 1),
            'uncertain': round(total_uncertain / count, 1),
            'explanation': {
                'signals': list(set(all_signals))[:15],
                'sources_analyzed': count
            }
        }

