from PyQt5.QtWidgets import QApplication
import logging
import re

# Set up a logger for this module
logger = logging.getLogger(__name__)

_SCALING_FACTOR = None

def get_scaling_factor():
    """
    Calculates and caches a scaling factor based on the primary screen's height.
    The reference height is 1080 pixels (standard Full HD).
    This function is lazy-loaded to ensure QApplication is initialized.
    """
    global _SCALING_FACTOR
    if _SCALING_FACTOR is not None:
        return _SCALING_FACTOR

    try:
        app = QApplication.instance()
        if not app:
            logger.warning("QApplication instance not found. Defaulting to scaling factor 1.0")
            _SCALING_FACTOR = 1.0
            return _SCALING_FACTOR

        screen = app.primaryScreen()
        if not screen:
            logger.warning("Primary screen not available, defaulting to scaling factor of 1.0.")
            _SCALING_FACTOR = 1.0
            return _SCALING_FACTOR
        
        reference_height = 1080.0
        actual_height = screen.geometry().height()
        
        # Calculate scaling factor
        scaling_factor = actual_height / reference_height
        
        # Clamp the scaling factor to a reasonable range to avoid extreme sizes
        clamped_factor = max(0.8, min(2.0, scaling_factor))
        
        logger.info(f"Screen height: {actual_height}px, Reference height: {reference_height}px, Scaling factor: {clamped_factor}")
        _SCALING_FACTOR = clamped_factor
        return _SCALING_FACTOR
        
    except Exception as e:
        logger.error(f"Error calculating scaling factor: {e}", exc_info=True)
        _SCALING_FACTOR = 1.0
        return 1.0

def scale_font(font_size_pt):
    """
    Scales a font size in points based on the global scaling factor.
    Returns an integer value for the new font size.
    """
    scaled_size = int(font_size_pt * get_scaling_factor())
    return scaled_size

def scale_widget_size(size_px):
    """
    Scales a widget dimension (e.g., height, width, padding) in pixels.
    Returns an integer value for the new size.
    """
    scaled_size = int(size_px * get_scaling_factor())
    return scaled_size

def get_adaptive_stylesheet(base_stylesheet):
    """
    Dynamically adjusts font sizes and pixel dimensions in a stylesheet string.
    Example input: "font-size: 10pt; padding: 8px;"
    Example output: "font-size: 12pt; padding: 10px;" (if scaled)
    """
    
    def replace_font_size(match):
        val = int(match.group(1))
        return f"font-size: {scale_font(val)}pt"
        
    def replace_pixel_size(match):
        prop = match.group(1)
        val = int(match.group(2))
        return f"{prop}: {scale_widget_size(val)}px"

    # Regex to find font-size in points (pt)
    stylesheet = re.sub(r"font-size:\s*(\d+)\s*pt", replace_font_size, base_stylesheet)
    
    # Regex to find padding, margin, border-radius in pixels (px)
    stylesheet = re.sub(r"(padding|margin|border-radius|margin-bottom):\s*(\d+)\s*px", replace_pixel_size, stylesheet)
    
    return stylesheet
