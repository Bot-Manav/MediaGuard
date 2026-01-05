"""
Module 5: Verification & Re-Upload Check
Checks if uploaded content matches protected fingerprints.
"""

from typing import Dict, Any, List, Optional
import hashlib


class VerificationEngine:
    """Verifies content against protected fingerprint database"""
    
    def __init__(self):
        # Similarity thresholds
        self.exact_match_threshold = 1.0  # 100% match
        self.perceptual_similarity_threshold = 0.85  # 85% similarity
        self.face_match_threshold = 0.90  # 90% face similarity
    
    def verify(self, 
              fingerprint_data: Dict[str, Any],
              protected_fingerprints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify if fingerprint matches any protected content.
        
        Returns:
            {
                'match_found': bool,
                'match_type': str ('exact' | 'perceptual' | 'face' | None),
                'similarity': float (0-100),
                'matched_fingerprint': Optional[Dict],
                'details': Dict
            }
        """
        if not protected_fingerprints:
            return {
                'match_found': False,
                'match_type': None,
                'similarity': 0.0,
                'matched_fingerprint': None,
                'details': {'message': 'No protected fingerprints in database'}
            }
        
        best_match = None
        best_similarity = 0.0
        best_match_type = None
        
        for protected in protected_fingerprints:
            protected_fp = protected.get('fingerprint', {})
            
            # 1. Check exact match (SHA-256)
            sha256_match = self._check_exact_match(
                fingerprint_data.get('sha256'),
                protected_fp.get('sha256')
            )
            
            if sha256_match:
                return {
                    'match_found': True,
                    'match_type': 'exact',
                    'similarity': 100.0,
                    'matched_fingerprint': protected,
                    'details': {
                        'method': 'SHA-256 cryptographic hash match',
                        'confidence': '100%'
                    }
                }
            
            # 2. Check perceptual similarity
            perceptual_sim = self._check_perceptual_similarity(
                fingerprint_data.get('perceptual_hash'),
                protected_fp.get('perceptual_hash')
            )
            
            if perceptual_sim > best_similarity:
                best_similarity = perceptual_sim
                best_match = protected
                best_match_type = 'perceptual'
            
            # 3. Check face embedding similarity (if available)
            if fingerprint_data.get('face_embedding_hash') and protected_fp.get('face_embedding_hash'):
                face_sim = self._check_face_similarity(
                    fingerprint_data.get('face_embedding_hash'),
                    protected_fp.get('face_embedding_hash')
                )
                
                if face_sim > best_similarity:
                    best_similarity = face_sim
                    best_match = protected
                    best_match_type = 'face'
        
        # Determine if match is significant
        if best_similarity >= (self.perceptual_similarity_threshold * 100):
            return {
                'match_found': True,
                'match_type': best_match_type,
                'similarity': best_similarity,
                'matched_fingerprint': best_match,
                'details': {
                    'method': f'{best_match_type} similarity match',
                    'confidence': f'{best_similarity:.1f}%'
                }
            }
        else:
            return {
                'match_found': False,
                'match_type': None,
                'similarity': best_similarity,
                'matched_fingerprint': None,
                'details': {
                    'message': 'No significant match found',
                    'highest_similarity': f'{best_similarity:.1f}%'
                }
            }
    
    def _check_exact_match(self, hash1: Optional[str], hash2: Optional[str]) -> bool:
        """Check if two hashes are exactly the same"""
        if not hash1 or not hash2:
            return False
        return hash1.lower() == hash2.lower()
    
    def _check_perceptual_similarity(self, 
                                    hash1: Optional[str], 
                                    hash2: Optional[str]) -> float:
        """
        Calculate perceptual hash similarity (0-100).
        Uses Hamming distance for imagehash, or string similarity for others.
        """
        if not hash1 or not hash2:
            return 0.0
        
        # If both are imagehash format (hex strings)
        try:
            # Try to interpret as imagehash (hex)
            if len(hash1) == 16 and len(hash2) == 16:
                # Calculate Hamming distance
                hamming = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
                similarity = (1 - hamming / len(hash1)) * 100
                return max(0.0, similarity)
        except Exception:
            pass
        
        # Fallback: string similarity
        if hash1 == hash2:
            return 100.0
        
        # Simple character-based similarity
        common_chars = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        similarity = (common_chars / max(len(hash1), len(hash2))) * 100
        return similarity
    
    def _check_face_similarity(self, 
                              hash1: Optional[str], 
                              hash2: Optional[str]) -> float:
        """
        Calculate face embedding similarity (0-100).
        MVP: Simplified - in production would use actual face embedding distance.
        """
        if not hash1 or not hash2:
            return 0.0
        
        if hash1 == hash2:
            return 100.0
        
        # Simplified similarity for MVP
        common_chars = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        similarity = (common_chars / max(len(hash1), len(hash2))) * 100
        return similarity

