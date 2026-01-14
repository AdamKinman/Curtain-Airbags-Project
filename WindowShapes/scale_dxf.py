"""
Scale DXF files based on estimated real-world dimensions from car images.

This module uses the Gemini API to estimate the dimensions of side windows
in car images, then creates scaled DXF files where 1 pixel = 1 millimeter.
"""

import os
import re
import json
import ezdxf
from google import genai
from google.genai import types
from typing import List, Optional

from pathlib import Path

from config import (
    ORIGINAL_IMAGE_FOLDER,
    DXF_FOLDER,
    SCALED_DXF_FOLDER
)

# Load GEMINI_API_KEY from a local .env file

def _load_dotenv_local(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

_load_dotenv_local(Path(__file__).with_name(".env"))

API_KEY = os.getenv("GEMINI_API_KEY")


def get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Define it in a .env file in the WindowShapes folder "
            "or in the environment."
        )
    return api_key


def create_client(api_key: Optional[str] = None) -> genai.Client:
    return genai.Client(api_key=api_key or get_api_key())

# Prompt template for Gemini to estimate window dimensions
DIMENSION_PROMPT = """
What are the dimensions of the side windows of the car in this image?
At the end of your reply, include a final estimation of the full combined width and height of the side windows in JSON format. That is, the estimated width should be of the full distance between the leftmost and rightmost edges of the side windows, and the height should be of the full distance between the topmost and bottommost edges of the side windows. If there are multiple side windows, DO NOT include the dimensions of each individual window, ONLY the combined full width and height of all side windows together.
Use this format:
{
  "width": <estimated full combined width of side windows in mm>,
  "height": <estimated full combined height of side windows in mm>
}
"""


def load_image(image_path: str) -> types.Part:
    """
    Load an image file and return it as a Gemini-compatible Part object.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        types.Part object containing the image data
    """
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)


def estimate_dimensions(client: genai.Client, image_path: str):
    """
    Use Gemini API to estimate the dimensions of side windows in a car image.
    
    Args:
        client: Gemini API client
        image_path: Path to the car image
    
    Returns:
        Tuple of (width_mm, height_mm) estimated dimensions in millimeters,
        or None if estimation fails
    """
    image = load_image(image_path)
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[DIMENSION_PROMPT, image],
    )
    
    response_text = response.text
    
    # Extract JSON from the response
    json_match = re.search(r'\{[^}]*"width"[^}]*"height"[^}]*\}', response_text, re.DOTALL)
    if not json_match:
        # Try alternative pattern
        json_match = re.search(r'\{[^}]*\}', response_text, re.DOTALL)
    
    if json_match:
        try:
            dimensions = json.loads(json_match.group())
            width = float(dimensions.get("width", 0))
            height = float(dimensions.get("height", 0))
            if width > 0 and height > 0:
                return (width, height)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    
    print(f"Warning: Could not parse dimensions from response for {image_path}")
    return None


def get_dxf_bounding_box(dxf_path: str):
    """
    Calculate the bounding box of all entities in a DXF file.
    
    Args:
        dxf_path: Path to the DXF file
    
    Returns:
        Tuple of (minX, minY, maxX, maxY) representing the bounding box in pixels,
        or None if the file contains no valid entities
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    
    has_entities = False
    
    for entity in msp:
        if entity.dxftype() == 'LWPOLYLINE':
            for x, y, *_ in entity.get_points():
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                has_entities = True
    
    if not has_entities:
        return None
    
    return (min_x, min_y, max_x, max_y)


def calculate_scale_factor(dxf_bounding_box, estimated_dimensions_mm):
    """
    Calculate the scale factor to convert DXF coordinates to millimeters.
    
    The scale factor is calculated so that the DXF dimensions match the
    estimated real-world dimensions. Uses the average of width and height
    scale factors for uniform scaling.
    
    Args:
        dxf_bounding_box: Tuple of (minX, minY, maxX, maxY) in pixels
        estimated_dimensions_mm: Tuple of (width_mm, height_mm) in millimeters
    
    Returns:
        Scale factor (multiply DXF coordinates by this value to get mm)
    """
    min_x, min_y, max_x, max_y = dxf_bounding_box
    width_mm, height_mm = estimated_dimensions_mm
    
    dxf_width = max_x - min_x
    dxf_height = max_y - min_y
    
    # Calculate scale factors for width and height
    scale_x = width_mm / dxf_width if dxf_width > 0 else 1.0
    scale_y = height_mm / dxf_height if dxf_height > 0 else 1.0
    
    # Use average scale factor for uniform scaling
    scale_factor = (scale_x + scale_y) / 2.0
    return scale_factor


def create_scaled_dxf(input_dxf_path: str, output_dxf_path: str, scale_factor: float) -> None:
    """
    Create a scaled version of a DXF file.
    
    All coordinates are multiplied by the scale factor so that
    1 unit in the output DXF corresponds to 1 millimeter.
    
    Args:
        input_dxf_path: Path to the source DXF file
        output_dxf_path: Path for the scaled output DXF file
        scale_factor: Factor to multiply all coordinates by
    """
    doc = ezdxf.readfile(input_dxf_path)
    msp = doc.modelspace()
    
    # Scale all LWPOLYLINE entities
    for entity in msp:
        if entity.dxftype() == 'LWPOLYLINE':
            # Get original points and scale them
            original_points = list(entity.get_points())
            scaled_points = [(x * scale_factor, y * scale_factor) for x, y, *_ in original_points]
            
            # Clear and set new points
            entity.set_points(scaled_points)
    
    # Save the scaled DXF
    doc.saveas(output_dxf_path)


def process_image(client: genai.Client, image_file_name: str) -> bool:
    """
    Process a single car image: estimate dimensions and create scaled DXF.
    
    Args:
        client: Gemini API client
        image_file_name: Name of the image file (e.g., "A_Fia_03.jpg")
    
    Returns:
        True if processing succeeded, False otherwise
    """
    base_name = os.path.splitext(image_file_name)[0]
    image_path = os.path.join(ORIGINAL_IMAGE_FOLDER, image_file_name)
    input_dxf_path = os.path.join(DXF_FOLDER, f"{base_name}.dxf")
    output_dxf_path = os.path.join(SCALED_DXF_FOLDER, f"{base_name}.dxf")
    
    # Check if corresponding DXF exists
    if not os.path.exists(input_dxf_path):
        print(f"Skipping {image_file_name}: No corresponding DXF file found")
        return False
    
    print(f"Processing {image_file_name}...")
    
    # Estimate dimensions using Gemini
    dimensions = estimate_dimensions(client, image_path)
    if dimensions is None:
        print(f"  Failed to estimate dimensions")
        return False
    
    width_mm, height_mm = dimensions
    print(f"  Estimated dimensions: {width_mm:.0f}mm x {height_mm:.0f}mm")
    
    # Get DXF bounding box
    bounding_box = get_dxf_bounding_box(input_dxf_path)
    if bounding_box is None:
        print(f"  Failed to read DXF bounding box")
        return False
    
    min_x, min_y, max_x, max_y = bounding_box
    dxf_width = max_x - min_x
    dxf_height = abs(max_y - min_y)  # abs() because Y might be flipped
    print(f"  DXF dimensions: {dxf_width:.0f}px x {dxf_height:.0f}px")
    
    # Calculate scale factor
    scale_factor = calculate_scale_factor(bounding_box, dimensions)
    print(f"  Scale factor: {scale_factor:.4f} (1px = {scale_factor:.4f}mm)")
    
    # Create scaled DXF
    create_scaled_dxf(input_dxf_path, output_dxf_path, scale_factor)
    print(f"  Saved scaled DXF to: {output_dxf_path}")
    
    return True


def run(image_file_names: List[str]) -> None:
    """
    Process car images and create scaled DXF files.
    
    Args:
        image_file_names: List of image file names to process.
    """
    if not image_file_names:
        raise ValueError("No image files provided")

    os.makedirs(SCALED_DXF_FOLDER, exist_ok=True)
    
    # Initialize Gemini client
    client = create_client()

    success_count = 0
    fail_count = 0

    for image_file_name in image_file_names:
        try:
            if process_image(client, image_file_name):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"Error processing {image_file_name}: {e}")
            fail_count += 1
    
    print(f"\nCompleted: {success_count} succeeded, {fail_count} failed")


# Backwards-compatible aliases (deprecated)
loadImage = load_image
estimateDimensions = estimate_dimensions
getDxfBoundingBox = get_dxf_bounding_box
calculateScaleFactor = calculate_scale_factor
createScaledDxf = create_scaled_dxf
processImage = process_image


if __name__ == "__main__":
    # Example: Process a single image
    run(["A_Fia_03.jpg"])