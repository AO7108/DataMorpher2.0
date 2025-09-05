import logging
from typing import Optional

# Dependency check for deepface
try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except ImportError:
    HAS_DEEPFACE = False

logger = logging.getLogger(__name__)

def get_expression(image_path: str) -> Optional[str]:
    """
    Analyzes an image using a robust deepface configuration.
    Returns 'smiling' or 'non smiling', or None if no face is detected.
    """
    if not HAS_DEEPFACE:
        logger.warning("DeepFace library not found. Cannot perform expression analysis.")
        return None

    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            # NEW: Use a more accurate and modern face detector
            detector_backend='retinaface',
            enforce_detection=True
        )
        dominant_emotion = result[0]['dominant_emotion']
        
        # NEW: More robust logic. 'happy' is smiling, everything else is non smiling.
        if dominant_emotion == 'happy':
            return 'smiling'
        else:
            return 'non smiling'
            
    except Exception:
        # This will catch errors from DeepFace, most commonly when no face is detected.
        return None