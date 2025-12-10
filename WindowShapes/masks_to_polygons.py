import torch
import numpy as np
import cv2
import os

from config import (
    PROCESSED_MASK_FOLDER as MASK_FOLDER,
    POLYGON_FOLDER as OUTPUT_FOLDER,
    POLYGON_CORNERS
)

'''
Output format:
x1 y1 x2 y2 x3 y3 ... xn yn (each x, y pair is a corner of a polygon sorted counter-clockwise,
                             each line contains one polygon)
'''


def loadProcessedMasks(baseFileName):
    """
    Load processed masks from the MASK_FOLDER.
    
    Args:
        baseFileName: Base name of the mask file (without extension)
    
    Returns:
        List of 2D boolean tensors, one for each mask
    """
    masks = torch.load(os.path.join(MASK_FOLDER, baseFileName + ".pt"))
    
    # Handle different possible shapes - return list of 2D masks
    if len(masks.shape) == 4:
        # Shape: (N, C, H, W) - extract each mask
        return [masks[i, 0] > 0 for i in range(masks.shape[0])]
    elif len(masks.shape) == 3:
        # Shape: (N, H, W) - each slice is a mask
        return [masks[i] > 0 for i in range(masks.shape[0])]
    elif len(masks.shape) == 2:
        # Single 2D mask
        return [masks > 0]
    return [masks]


def maskToPolygons(mask):
    assert len(mask.shape) == 2, f"Mask should be a 2D Tensor, got shape {[s for s in mask.shape]}"

    mask_np = mask.cpu().numpy().astype(np.uint8)
    mask_cv = (mask_np * 255).astype(np.uint8) # Ecpected format by cv2

    contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [contourToPolygon(c) for c in contours]
    return [polygon[:, 0, :].flatten().tolist() for polygon in polygons] # Flatten to x1, y1, x2, y2, ...

def contourToPolygon(contour):
    """
    Convert a contour to a polygon with exactly POLYGON_CORNERS using the Ramer-Douglas-Peucker algorithm.
    Uses binary search to find the optimal epsilon value that yields the desired number of corners.
    """
    # Get the perimeter for epsilon scaling
    perimeter = cv2.arcLength(contour, closed=True)
    
    # Binary search for the right epsilon
    eps_low, eps_high = 0.0, 0.1
    best_polygon = None
    best_diff = float('inf')
    
    # First, find an upper bound where we get fewer points than POLYGON_CORNERS
    while True:
        polygon = cv2.approxPolyDP(contour, eps_high * perimeter, closed=True)
        if len(polygon) <= POLYGON_CORNERS:
            break
        eps_high *= 2
        if eps_high > 1.0:  # Safety limit
            break
    
    # Binary search to find epsilon that gives exactly POLYGON_CORNERS
    for _ in range(100):  # Max iterations
        eps_mid = (eps_low + eps_high) / 2
        polygon = cv2.approxPolyDP(contour, eps_mid * perimeter, closed=True)
        n_points = len(polygon)
        
        # Track the best result closest to POLYGON_CORNERS
        diff = abs(n_points - POLYGON_CORNERS)
        if diff < best_diff or (diff == best_diff and n_points == POLYGON_CORNERS):
            best_diff = diff
            best_polygon = polygon
        
        if n_points == POLYGON_CORNERS:
            return polygon
        elif n_points > POLYGON_CORNERS:
            eps_low = eps_mid  # Need more simplification
        else:
            eps_high = eps_mid  # Need less simplification
        
        # Convergence check
        if eps_high - eps_low < 1e-10:
            break
    
    return best_polygon


def run():
    mask_files = [f for f in os.listdir(MASK_FOLDER) if f.endswith('.pt')]
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for mask_file in mask_files:
        baseFileName = mask_file.split('.')[0]
        try:
            masks = loadProcessedMasks(baseFileName)
            all_polygons = []
            for mask in masks:
                polygons = maskToPolygons(mask)
                all_polygons.extend(polygons)
            with open(os.path.join(OUTPUT_FOLDER, f"{baseFileName}.txt"), 'w') as f:
                for polygon in all_polygons:
                    f.write(' '.join(map(str, polygon)) + '\n')
        except Exception as e:
            print(f"Error processing {mask_file}: {e}")


if __name__ == "__main__":
    run()