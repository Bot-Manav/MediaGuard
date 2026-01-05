"""
Module 1: Media Upload & Secure Intake
Handles secure, temporary processing of uploaded media files.
No long-term storage - files are processed in memory and deleted immediately.
"""

import io
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import streamlit as st


class MediaUploadHandler:
    """Handles secure media upload and temporary processing"""
    
    def __init__(self):
        self.temp_files = []
    
    def process_upload(self, uploaded_file) -> Optional[Dict[str, Any]]:
        """
        Process uploaded file securely.
        Returns media data dictionary or None if processing fails.
        """
        try:
            # Read file into memory
            file_bytes = uploaded_file.read()
            
            # Validate file size (max 10MB)
            max_size = 10 * 1024 * 1024  # 10MB
            if len(file_bytes) > max_size:
                st.error(f"File too large. Maximum size: 10MB")
                return None
            
            # Create temporary file for processing
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_file.name).suffix
            )
            temp_file.write(file_bytes)
            temp_file.close()
            
            self.temp_files.append(temp_file.name)
            
            # Load image with PIL for validation
            try:
                image = Image.open(io.BytesIO(file_bytes))
                image.verify()  # Verify it's a valid image
                
                # Reopen for actual use (verify() closes the file)
                image = Image.open(io.BytesIO(file_bytes))
                
                # Get image metadata
                width, height = image.size
                format_name = image.format or "UNKNOWN"
                mode = image.mode
                
            except Exception as e:
                st.warning(f"Image validation warning: {str(e)}")
                image = Image.open(io.BytesIO(file_bytes))
                width, height = image.size
                format_name = image.format or "UNKNOWN"
                mode = image.mode
            
            # Return media data (file path for processing, bytes for hashing)
            return {
                'file_path': temp_file.name,
                'file_bytes': file_bytes,
                'filename': uploaded_file.name,
                'size': len(file_bytes),
                'width': width,
                'height': height,
                'format': format_name,
                'mode': mode,
                'image': image
            }
            
        except Exception as e:
            st.error(f"Error processing upload: {str(e)}")
            return None
    
    def cleanup(self, media_data: Optional[Dict[str, Any]]):
        """Clean up temporary files"""
        if media_data and 'file_path' in media_data:
            try:
                if Path(media_data['file_path']).exists():
                    Path(media_data['file_path']).unlink()
            except Exception:
                pass  # File may already be deleted
        
        # Clean up any remaining temp files
        for temp_file in self.temp_files:
            try:
                if Path(temp_file).exists():
                    Path(temp_file).unlink()
            except Exception:
                pass
        
        self.temp_files.clear()
        
        # Clear image from memory if present
        if media_data and 'image' in media_data:
            try:
                media_data['image'].close()
            except Exception:
                pass

