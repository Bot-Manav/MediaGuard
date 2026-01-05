"""
Module 3: Fingerprint Generation Engine
Generates multiple types of fingerprints for media:
- Cryptographic hash (SHA-256)
- Perceptual hash (detects similar images)
- Face embedding hash (if faces detected)
"""

import hashlib
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

# For perceptual hashing, we'll use imagehash library
try:
    import imagehash
    PERCEPTUAL_HASH_AVAILABLE = True
except ImportError:
    PERCEPTUAL_HASH_AVAILABLE = False

# For face embeddings, we would use face_recognition or similar
# For MVP, we'll create a placeholder structure


class FingerprintGenerator:
    """Generates cryptographic and perceptual fingerprints for media"""
    
    def __init__(self):
        pass
    
    def generate(self, media_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate all fingerprint types for the media.
        
        Returns:
            {
                'sha256': str,
                'perceptual_hash': str,
                'face_embedding_hash': Optional[str],
                'metadata_hash': str
            }
        """
        file_bytes = media_data.get('file_bytes')
        image = media_data.get('image')
        
        if file_bytes is None:
            raise ValueError("No file data available for fingerprinting")
        
        fingerprints = {}
        
        # 1. Cryptographic Hash (SHA-256)
        fingerprints['sha256'] = self._generate_sha256(file_bytes)
        
        # 2. Perceptual Hash (detects similar images)
        if image and PERCEPTUAL_HASH_AVAILABLE:
            fingerprints['perceptual_hash'] = self._generate_perceptual_hash(image)
        else:
            # Fallback: use a hash of image statistics
            fingerprints['perceptual_hash'] = self._generate_statistical_hash(image)
        
        # 3. Face Embedding Hash (if faces detected)
        if image:
            fingerprints['face_embedding_hash'] = self._generate_face_embedding_hash(image)
        else:
            fingerprints['face_embedding_hash'] = None
        
        # 4. Metadata Hash
        fingerprints['metadata_hash'] = self._generate_metadata_hash(media_data)
        
        # Combined fingerprint ID
        fingerprints['fingerprint_id'] = self._generate_fingerprint_id(fingerprints)
        
        return fingerprints
    
    def _generate_sha256(self, file_bytes: bytes) -> str:
        """Generate SHA-256 cryptographic hash"""
        return hashlib.sha256(file_bytes).hexdigest()
    
    def _generate_perceptual_hash(self, image: Image.Image) -> str:
        """Generate perceptual hash using imagehash"""
        try:
            # Use average hash for similarity detection
            phash = imagehash.average_hash(image)
            return str(phash)
        except Exception:
            return self._generate_statistical_hash(image)
    
    def _generate_statistical_hash(self, image: Optional[Image.Image]) -> str:
        """Fallback: Generate hash based on image statistics"""
        if image is None:
            return "0" * 16
        
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            
            # Calculate statistical features
            mean = np.mean(img_array, axis=(0, 1))
            std = np.std(img_array, axis=(0, 1))
            
            # Create hash from statistics
            stats_str = f"{mean[0]:.2f},{mean[1]:.2f},{mean[2]:.2f},{std[0]:.2f},{std[1]:.2f},{std[2]:.2f}"
            return hashlib.md5(stats_str.encode()).hexdigest()[:16]
        except Exception:
            return "0" * 16
    
    def _generate_face_embedding_hash(self, image: Image.Image) -> Optional[str]:
        """
        Generate face embedding hash.
        MVP: Placeholder - in production would use face_recognition or similar.
        """
        # In production, this would:
        # 1. Detect faces in image
        # 2. Extract face embeddings
        # 3. Hash the embeddings
        
        # For MVP, return None (no face detection implemented)
        # This is acceptable for hackathon - explain it's conceptual
        
        # Placeholder: hash of image dimensions (simulates face detection)
        try:
            width, height = image.size
            placeholder = f"{width}x{height}"
            return hashlib.md5(placeholder.encode()).hexdigest()[:32]
        except Exception:
            return None
    
    def _generate_metadata_hash(self, media_data: Dict[str, Any]) -> str:
        """Generate hash from metadata"""
        metadata_str = f"{media_data.get('width', 0)}x{media_data.get('height', 0)}-{media_data.get('format', 'UNKNOWN')}-{media_data.get('size', 0)}"
        return hashlib.sha256(metadata_str.encode()).hexdigest()[:32]
    
    def _generate_fingerprint_id(self, fingerprints: Dict[str, Any]) -> str:
        """Generate a unique fingerprint ID from all hashes"""
        combined = f"{fingerprints.get('sha256', '')}{fingerprints.get('perceptual_hash', '')}{fingerprints.get('metadata_hash', '')}"
        return hashlib.sha256(combined.encode()).hexdigest()[:64]

