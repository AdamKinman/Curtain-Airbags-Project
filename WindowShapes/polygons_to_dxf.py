import os
import ezdxf

from config import (
    POLYGON_FOLDER,
    DXF_FOLDER
)

'''
Input format (from polygon files):
x1 y1 x2 y2 x3 y3 ... xn yn (each x, y pair is a corner of a polygon sorted counter-clockwise,
                             each line contains one polygon)

Output:
DXF files with each polygon as a closed polyline
'''


def loadPolygons(filePath):
    """
    Load polygons from a text file.
    
    Args:
        filePath: Path to the polygon file
    
    Returns:
        List of polygons, where each polygon is a list of (x, y) tuples
    """
    polygons = []
    with open(filePath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            coords = list(map(float, line.split()))
            # Convert flat list to list of (x, y) tuples
            polygon = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            polygons.append(polygon)
    return polygons


def polygonsToDxf(polygons, outputPath):
    """
    Convert polygons to a DXF file.
    
    Args:
        polygons: List of polygons, where each polygon is a list of (x, y) tuples
        outputPath: Path to save the DXF file
    """
    # Create a new DXF document
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    
    for i, polygon in enumerate(polygons):
        if len(polygon) < 3:
            print(f"Warning: Skipping polygon {i} with less than 3 vertices")
            continue
        
        # Flip Y coordinates (image Y increases downward, DXF Y increases upward)
        flipped_polygon = [(x, -y) for x, y in polygon]
        
        # Add a closed polyline for each polygon
        # close=True ensures the polyline forms a closed shape
        msp.add_lwpolyline(flipped_polygon, close=True)
    
    # Save the DXF file
    doc.saveas(outputPath)


def run():
    """
    Convert all polygon files in the POLYGON_FOLDER to DXF files.
    """
    print("Converting polygons to DXF...")
    os.makedirs(DXF_FOLDER, exist_ok=True)
    
    polygon_files = [f for f in os.listdir(POLYGON_FOLDER) if f.endswith('.txt')]
    
    for polygon_file in polygon_files:
        baseFileName = os.path.splitext(polygon_file)[0]
        inputPath = os.path.join(POLYGON_FOLDER, polygon_file)
        outputPath = os.path.join(DXF_FOLDER, f"{baseFileName}.dxf")
        
        try:
            polygons = loadPolygons(inputPath)
            if polygons:
                polygonsToDxf(polygons, outputPath)
            else:
                print(f"Warning: No polygons found in {polygon_file}")
        except Exception as e:
            print(f"Error processing {polygon_file}: {e}")
    
    print(f"\nDXF files saved to: {DXF_FOLDER}")


if __name__ == "__main__":
    run()
