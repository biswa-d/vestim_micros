#!/usr/bin/env python3
"""
Fix PyBattML icon by creating a proper ICO file with solid white background
"""

try:
    from PIL import Image
    
    # Load the PNG image
    png_path = "vestim/gui/resources/PyBattML_icon.png"
    ico_path = "vestim/gui/resources/PyBattML_icon_fixed.ico"
    
    # Open the PNG image
    img = Image.open(png_path)
    print(f"Original image: {img.size}, mode: {img.mode}")
    
    # Create multiple sizes for the ICO file
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    
    # Create a list of resized images with solid white background
    icon_images = []
    for size in sizes:
        # Create a new image with solid white background
        white_bg = Image.new('RGB', size, 'white')
        
        # Resize the original image
        resized = img.resize(size, Image.Resampling.LANCZOS)
        
        # If the original has transparency, paste it onto white background
        if resized.mode == 'RGBA':
            white_bg.paste(resized, (0, 0), resized)
        else:
            white_bg.paste(resized, (0, 0))
        
        icon_images.append(white_bg)
        print(f"Created {size[0]}x{size[1]} version with white background")
    
    # Save as ICO with multiple sizes
    icon_images[0].save(
        ico_path,
        format='ICO',
        sizes=[(img.width, img.height) for img in icon_images],
        append_images=icon_images[1:]
    )
    
    print(f"✅ Created icon with solid white background: {ico_path}")
    print("Replace the original PyBattML_icon.ico with PyBattML_icon_fixed.ico")
    
except ImportError:
    print("❌ PIL (Pillow) not installed. Install with: pip install Pillow")
except Exception as e:
    print(f"❌ Error: {e}")