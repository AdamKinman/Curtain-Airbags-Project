"""Convert polygon text files into DXF polylines."""

import os

import ezdxf

from config import (
    POLYGON_FOLDER,
    DXF_FOLDER
)

def load_polygons(file_path: str):
    """
    Load polygons from a text file.
    
    Args:
        file_path: Path to the polygon file
    
    Returns:
        List of polygons, where each polygon is a list of (x, y) tuples
    """
    polygons = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            coords = list(map(float, line.split()))
            # Convert flat list to list of (x, y) tuples
            polygon = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            polygons.append(polygon)
    return polygons


def polygons_to_dxf(polygons, output_path: str) -> None:
    """
    Convert polygons to a DXF file.
    
    Args:
        polygons: List of polygons, where each polygon is a list of (x, y) tuples
        output_path: Path to save the DXF file
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
    doc.saveas(output_path)


def run():
    """
    Convert all polygon files in the POLYGON_FOLDER to DXF files.
    """
    print("Converting polygons to DXF...")
    os.makedirs(DXF_FOLDER, exist_ok=True)
    
    polygon_files = [f for f in os.listdir(POLYGON_FOLDER) if f.endswith('.txt')]
    
    for polygon_file in polygon_files:
        base_name = os.path.splitext(polygon_file)[0]
        input_path = os.path.join(POLYGON_FOLDER, polygon_file)
        output_path = os.path.join(DXF_FOLDER, f"{base_name}.dxf")
        
        try:
            polygons = load_polygons(input_path)
            if polygons:
                polygons_to_dxf(polygons, output_path)
            else:
                print(f"Warning: No polygons found in {polygon_file}")
        except Exception as e:
            print(f"Error processing {polygon_file}: {e}")
    
    print(f"\nDXF files saved to: {DXF_FOLDER}")


# Backwards-compatible aliases (deprecated)
loadPolygons = load_polygons
polygonsToDxf = polygons_to_dxf


if __name__ == "__main__":
    run()
