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

from config import (
    ORIGINAL_IMAGE_FOLDER,
    DXF_FOLDER,
    SCALED_DXF_FOLDER
)

# Gemini API key
API_KEY = "AIzaSyAk2F3GQejOap1ZVohr5REhE485No9bEHI"

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


def loadImage(imagePath):
    """
    Load an image file and return it as a Gemini-compatible Part object.
    
    Args:
        imagePath: Path to the image file
    
    Returns:
        types.Part object containing the image data
    """
    with open(imagePath, "rb") as f:
        imageBytes = f.read()
    return types.Part.from_bytes(data=imageBytes, mime_type="image/jpeg")


def estimateDimensions(client, imagePath):
    """
    Use Gemini API to estimate the dimensions of side windows in a car image.
    
    Args:
        client: Gemini API client
        imagePath: Path to the car image
    
    Returns:
        Tuple of (width_mm, height_mm) estimated dimensions in millimeters,
        or None if estimation fails
    """
    image = loadImage(imagePath)
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[DIMENSION_PROMPT, image],
    )
    
    responseText = response.text
    
    # Extract JSON from the response
    jsonMatch = re.search(r'\{[^}]*"width"[^}]*"height"[^}]*\}', responseText, re.DOTALL)
    if not jsonMatch:
        # Try alternative pattern
        jsonMatch = re.search(r'\{[^}]*\}', responseText, re.DOTALL)
    
    if jsonMatch:
        try:
            dimensions = json.loads(jsonMatch.group())
            width = float(dimensions.get("width", 0))
            height = float(dimensions.get("height", 0))
            if width > 0 and height > 0:
                return (width, height)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    
    print(f"Warning: Could not parse dimensions from response for {imagePath}")
    return None


def getDxfBoundingBox(dxfPath):
    """
    Calculate the bounding box of all entities in a DXF file.
    
    Args:
        dxfPath: Path to the DXF file
    
    Returns:
        Tuple of (minX, minY, maxX, maxY) representing the bounding box in pixels,
        or None if the file contains no valid entities
    """
    doc = ezdxf.readfile(dxfPath)
    msp = doc.modelspace()
    
    minX = float('inf')
    minY = float('inf')
    maxX = float('-inf')
    maxY = float('-inf')
    
    hasEntities = False
    
    for entity in msp:
        if entity.dxftype() == 'LWPOLYLINE':
            for x, y, *_ in entity.get_points():
                minX = min(minX, x)
                minY = min(minY, y)
                maxX = max(maxX, x)
                maxY = max(maxY, y)
                hasEntities = True
    
    if not hasEntities:
        return None
    
    return (minX, minY, maxX, maxY)


def calculateScaleFactor(dxfBoundingBox, estimatedDimensionsMm):
    """
    Calculate the scale factor to convert DXF coordinates to millimeters.
    
    The scale factor is calculated so that the DXF dimensions match the
    estimated real-world dimensions. Uses the average of width and height
    scale factors for uniform scaling.
    
    Args:
        dxfBoundingBox: Tuple of (minX, minY, maxX, maxY) in pixels
        estimatedDimensionsMm: Tuple of (width_mm, height_mm) in millimeters
    
    Returns:
        Scale factor (multiply DXF coordinates by this value to get mm)
    """
    minX, minY, maxX, maxY = dxfBoundingBox
    widthMm, heightMm = estimatedDimensionsMm
    
    dxfWidth = maxX - minX
    dxfHeight = maxY - minY
    
    # Calculate scale factors for width and height
    scaleX = widthMm / dxfWidth if dxfWidth > 0 else 1.0
    scaleY = heightMm / dxfHeight if dxfHeight > 0 else 1.0
    
    # Use average scale factor for uniform scaling
    scaleFactor = (scaleX + scaleY) / 2.0
    
    return scaleFactor


def createScaledDxf(inputDxfPath, outputDxfPath, scaleFactor):
    """
    Create a scaled version of a DXF file.
    
    All coordinates are multiplied by the scale factor so that
    1 unit in the output DXF corresponds to 1 millimeter.
    
    Args:
        inputDxfPath: Path to the source DXF file
        outputDxfPath: Path for the scaled output DXF file
        scaleFactor: Factor to multiply all coordinates by
    """
    doc = ezdxf.readfile(inputDxfPath)
    msp = doc.modelspace()
    
    # Scale all LWPOLYLINE entities
    for entity in msp:
        if entity.dxftype() == 'LWPOLYLINE':
            # Get original points and scale them
            originalPoints = list(entity.get_points())
            scaledPoints = [(x * scaleFactor, y * scaleFactor) for x, y, *_ in originalPoints]
            
            # Clear and set new points
            entity.set_points(scaledPoints)
    
    # Save the scaled DXF
    doc.saveas(outputDxfPath)


def processImage(client, imageFileName):
    """
    Process a single car image: estimate dimensions and create scaled DXF.
    
    Args:
        client: Gemini API client
        imageFileName: Name of the image file (e.g., "A_Fia_03.jpg")
    
    Returns:
        True if processing succeeded, False otherwise
    """
    baseName = os.path.splitext(imageFileName)[0]
    imagePath = os.path.join(ORIGINAL_IMAGE_FOLDER, imageFileName)
    inputDxfPath = os.path.join(DXF_FOLDER, f"{baseName}.dxf")
    outputDxfPath = os.path.join(SCALED_DXF_FOLDER, f"{baseName}.dxf")
    
    # Check if corresponding DXF exists
    if not os.path.exists(inputDxfPath):
        print(f"Skipping {imageFileName}: No corresponding DXF file found")
        return False
    
    print(f"Processing {imageFileName}...")
    
    # Estimate dimensions using Gemini
    dimensions = estimateDimensions(client, imagePath)
    if dimensions is None:
        print(f"  Failed to estimate dimensions")
        return False
    
    widthMm, heightMm = dimensions
    print(f"  Estimated dimensions: {widthMm:.0f}mm x {heightMm:.0f}mm")
    
    # Get DXF bounding box
    boundingBox = getDxfBoundingBox(inputDxfPath)
    if boundingBox is None:
        print(f"  Failed to read DXF bounding box")
        return False
    
    minX, minY, maxX, maxY = boundingBox
    dxfWidth = maxX - minX
    dxfHeight = abs(maxY - minY)  # abs() because Y might be flipped
    print(f"  DXF dimensions: {dxfWidth:.0f}px x {dxfHeight:.0f}px")
    
    # Calculate scale factor
    scaleFactor = calculateScaleFactor(boundingBox, dimensions)
    print(f"  Scale factor: {scaleFactor:.4f} (1px = {scaleFactor:.4f}mm)")
    
    # Create scaled DXF
    createScaledDxf(inputDxfPath, outputDxfPath, scaleFactor)
    print(f"  Saved scaled DXF to: {outputDxfPath}")
    
    return True


def run(imageFileNames=None):
    """
    Process car images and create scaled DXF files.
    
    Args:
        imageFileNames: List of image file names to process.
                       If None, processes all images with corresponding DXF files.
    """
    assert imageFileNames is not None, "Attempted to scale all files, which would lead to many API calls. If this is intended, then remove this line."
    assert len(imageFileNames) < 15, "Attempted to scale many files, which would lead to many API calls. If this is intended, then remove this line."

    os.makedirs(SCALED_DXF_FOLDER, exist_ok=True)
    
    # Initialize Gemini client
    client = genai.Client(api_key=API_KEY)
    
    # Get list of images to process
    if imageFileNames is None:
        # Find all images that have corresponding DXF files
        dxfFiles = set(os.path.splitext(f)[0] for f in os.listdir(DXF_FOLDER) if f.endswith('.dxf'))
        imageFileNames = [
            f for f in os.listdir(ORIGINAL_IMAGE_FOLDER)
            if f.lower().endswith('.jpg') and os.path.splitext(f)[0] in dxfFiles
        ]
    
    successCount = 0
    failCount = 0
    
    for imageFileName in imageFileNames:
        try:
            if processImage(client, imageFileName):
                successCount += 1
            else:
                failCount += 1
        except Exception as e:
            print(f"Error processing {imageFileName}: {e}")
            failCount += 1
    
    print(f"\nCompleted: {successCount} succeeded, {failCount} failed")


if __name__ == "__main__":
    # Example: Process a single image
    run(["A_Fia_03.jpg"])