# VEstim Common GUI Styles
# This module provides consistent styling for all VEstim GUI components

from PyQt5.QtWidgets import QApplication, QDesktopWidget
from PyQt5.QtCore import Qt

class VEstimStyles:
    """Common styles and utilities for VEstim GUI components."""
    
    # Color palette
    PRIMARY_GREEN = "#0b6337"
    SECONDARY_BLUE = "#2E86AB"
    LIGHT_GRAY = "#f8f9fa"
    MEDIUM_GRAY = "#6c757d"
    DARK_GRAY = "#343a40"
    WHITE = "#ffffff"
    ERROR_RED = "#dc3545"
    SUCCESS_GREEN = "#28a745"
    
    @staticmethod
    def get_responsive_sizing():
        """Get responsive sizing based on screen resolution."""
        screen = QApplication.desktop().screenGeometry()
        width = screen.width()
        height = screen.height()
        
        # Base scaling factors
        if width >= 2560:  # 4K and higher
            scale_factor = 1.4
        elif width >= 1920:  # 1080p
            scale_factor = 1.2
        elif width >= 1366:  # 720p
            scale_factor = 1.0
        else:  # Lower resolutions
            scale_factor = 0.9
            
        return {
            'scale_factor': scale_factor,
            'window_width': min(int(width * 0.8), 1400),
            'window_height': min(int(height * 0.8), 900),
            'button_height': int(35 * scale_factor),
            'button_width_small': int(150 * scale_factor),
            'button_width_medium': int(200 * scale_factor),
            'button_width_large': int(250 * scale_factor),
            'font_size_small': max(9, int(9 * scale_factor)),
            'font_size_normal': max(10, int(10 * scale_factor)),
            'font_size_large': max(12, int(12 * scale_factor)),
            'font_size_title': max(16, int(16 * scale_factor)),
            'spacing_small': int(5 * scale_factor),
            'spacing_medium': int(10 * scale_factor),
            'spacing_large': int(15 * scale_factor),
            'margin': int(10 * scale_factor)
        }
    
    @staticmethod
    def get_button_style(button_type="primary", size="medium"):
        """Get consistent button styling."""
        sizing = VEstimStyles.get_responsive_sizing()
        
        base_style = f"""
            QPushButton {{
                font-size: {sizing['font_size_normal']}pt;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                min-height: {sizing['button_height']}px;
            }}
            QPushButton:hover {{
                opacity: 0.9;
            }}
            QPushButton:pressed {{
                opacity: 0.8;
            }}
            QPushButton:disabled {{
                opacity: 0.5;
            }}
        """
        
        if button_type == "primary":
            color_style = f"""
                background-color: {VEstimStyles.PRIMARY_GREEN};
                color: {VEstimStyles.WHITE};
            """
        elif button_type == "secondary":
            color_style = f"""
                background-color: {VEstimStyles.SECONDARY_BLUE};
                color: {VEstimStyles.WHITE};
            """
        elif button_type == "danger":
            color_style = f"""
                background-color: {VEstimStyles.ERROR_RED};
                color: {VEstimStyles.WHITE};
            """
        elif button_type == "outline":
            color_style = f"""
                background-color: transparent;
                color: {VEstimStyles.PRIMARY_GREEN};
                border: 2px solid {VEstimStyles.PRIMARY_GREEN};
            """
        else:  # default
            color_style = f"""
                background-color: {VEstimStyles.LIGHT_GRAY};
                color: {VEstimStyles.DARK_GRAY};
                border: 1px solid {VEstimStyles.MEDIUM_GRAY};
            """
            
        return base_style + color_style
    
    @staticmethod
    def get_title_style():
        """Get consistent title styling."""
        sizing = VEstimStyles.get_responsive_sizing()
        return f"""
            font-size: {sizing['font_size_title']}pt;
            font-weight: bold;
            color: {VEstimStyles.PRIMARY_GREEN};
            margin-bottom: {sizing['spacing_large']}px;
        """
    
    @staticmethod
    def get_subtitle_style():
        """Get consistent subtitle styling."""
        sizing = VEstimStyles.get_responsive_sizing()
        return f"""
            font-size: {sizing['font_size_large']}pt;
            font-weight: bold;
            color: {VEstimStyles.DARK_GRAY};
            margin-bottom: {sizing['spacing_medium']}px;
        """
    
    @staticmethod
    def get_group_box_style():
        """Get consistent group box styling."""
        sizing = VEstimStyles.get_responsive_sizing()
        return f"""
            QGroupBox {{
                font-size: {sizing['font_size_normal']}pt;
                font-weight: bold;
                border: 2px solid {VEstimStyles.MEDIUM_GRAY};
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: {sizing['spacing_medium']}px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: {sizing['spacing_medium']}px;
                padding: 0 5px 0 5px;
                color: {VEstimStyles.PRIMARY_GREEN};
            }}
        """
    
    @staticmethod
    def get_scroll_area_style():
        """Get consistent scroll area styling."""
        return f"""
            QScrollArea {{
                border: 1px solid {VEstimStyles.MEDIUM_GRAY};
                border-radius: 6px;
                background-color: {VEstimStyles.WHITE};
            }}
            QScrollBar:vertical {{
                background-color: {VEstimStyles.LIGHT_GRAY};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {VEstimStyles.MEDIUM_GRAY};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {VEstimStyles.PRIMARY_GREEN};
            }}
        """
    
    @staticmethod
    def get_input_style():
        """Get consistent input field styling."""
        sizing = VEstimStyles.get_responsive_sizing()
        return f"""
            QLineEdit, QComboBox, QTextEdit {{
                font-size: {sizing['font_size_normal']}pt;
                padding: 6px;
                border: 1px solid {VEstimStyles.MEDIUM_GRAY};
                border-radius: 4px;
                background-color: {VEstimStyles.WHITE};
            }}
            QLineEdit:focus, QComboBox:focus, QTextEdit:focus {{
                border-color: {VEstimStyles.PRIMARY_GREEN};
                outline: none;
            }}
            QListWidget {{
                font-size: {sizing['font_size_normal']}pt;
                border: 1px solid {VEstimStyles.MEDIUM_GRAY};
                border-radius: 4px;
                background-color: {VEstimStyles.WHITE};
                selection-background-color: {VEstimStyles.PRIMARY_GREEN};
            }}
        """
    
    @staticmethod
    def get_instruction_style():
        """Get consistent instruction text styling."""
        sizing = VEstimStyles.get_responsive_sizing()
        return f"""
            font-size: {sizing['font_size_small']}pt;
            color: {VEstimStyles.MEDIUM_GRAY};
            margin-bottom: {sizing['spacing_medium']}px;
            line-height: 1.4;
        """

    @staticmethod
    def apply_button_sizing(button, size="medium"):
        """Apply consistent button sizing."""
        sizing = VEstimStyles.get_responsive_sizing()
        
        if size == "small":
            button.setFixedWidth(sizing['button_width_small'])
        elif size == "large":
            button.setFixedWidth(sizing['button_width_large'])
        else:  # medium
            button.setFixedWidth(sizing['button_width_medium'])
            
        button.setFixedHeight(sizing['button_height'])
