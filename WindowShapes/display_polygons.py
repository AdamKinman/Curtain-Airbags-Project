"""Visualize polygon overlays on images."""

import os
from random import shuffle

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from image_files import list_image_files

from config import (
    HORIZONTAL_ALIGN_IMAGES as IMAGE_FOLDER,
    POLYGON_FOLDER
)

def parse_polygon_file(filepath):
    """Parse a polygon file and return a list of polygons.
    Each polygon is a list of (x, y) tuples."""
    polygons = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = list(map(int, line.split()))
            # Pair up consecutive values as (x, y) coordinates
            polygon = [(values[i], values[i+1]) for i in range(0, len(values), 2)]
            polygons.append(polygon)
    return polygons


def visualize_polygons_on_image(image_name: str) -> None:
    """Display an image with polygons overlaid."""
    # Load the image
    image = Image.open(os.path.join(IMAGE_FOLDER, image_name))
    # Convert to RGBA to support transparency
    image = image.convert("RGBA")
    
    # Load the polygons
    polygon_file = os.path.join(POLYGON_FOLDER, f"{os.path.splitext(image_name)[0]}.txt")
    polygons = parse_polygon_file(polygon_file)
    
    # Create a blank overlay image for drawing polygons
    polygon_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(polygon_overlay)
    
    # Draw each polygon
    for polygon in polygons:
        if len(polygon) < 3:
            continue  # Skip invalid polygons
        
        # Outline color: solid red
        outline_color = (255, 0, 0, 255)
        
        # Draw polygon outline with thicker lines
        # PIL's polygon() doesn't support width, so we use line()
        draw.line(polygon + [polygon[0]], fill=outline_color, width=7)
    
    # Composite the polygon overlay onto the original image
    result_image = Image.alpha_composite(image, polygon_overlay)
    
    # Display the result
    plt.imshow(result_image)
    plt.axis('off')
    plt.show()


# Backwards-compatible alias (deprecated)
visualizePolygonsOnImage = visualize_polygons_on_image


if __name__ == "__main__":
    files = list_image_files(IMAGE_FOLDER)
    shuffle(files)
    for file in files:
        polygon_path = os.path.join(POLYGON_FOLDER, f"{file.split('.')[0]}.txt")
        if os.path.exists(polygon_path):
            visualize_polygons_on_image(file)
