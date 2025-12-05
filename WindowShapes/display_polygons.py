import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
from random import shuffle

from config import (
    HORIZONTAL_ALIGN_IMAGES as IMAGE_FOLDER,
    POLYGON_FOLDER
)

'''
Polygon file format:
x1 y1 x2 y2 x3 y3 ... xn yn (each x, y pair is a corner of a polygon sorted counter-clockwise,
                             each line contains one polygon)
'''


def parse_polygon_file(filepath):
    """Parse a polygon file and return a list of polygons.
    Each polygon is a list of (x, y) tuples."""
    polygons = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = list(map(int, line.split()))
            # Pair up consecutive values as (x, y) coordinates
            polygon = [(values[i], values[i+1]) for i in range(0, len(values), 2)]
            polygons.append(polygon)
    return polygons


def visualizePolygonsOnImage(imageName):
    """Display an image with polygons overlaid."""
    # Load the image
    image = Image.open(os.path.join(IMAGE_FOLDER, imageName))
    # Convert to RGBA to support transparency
    image = image.convert("RGBA")
    
    # Load the polygons
    polygon_file = os.path.join(POLYGON_FOLDER, f"{imageName.split('.')[0]}.txt")
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


if __name__ == '__main__':
    files = os.listdir(IMAGE_FOLDER)
    shuffle(files)
    for file in files:
        if not file.lower().endswith('.jpg'):
            continue
        polygon_path = os.path.join(POLYGON_FOLDER, f"{file.split('.')[0]}.txt")
        if os.path.exists(polygon_path):
            visualizePolygonsOnImage(file)
