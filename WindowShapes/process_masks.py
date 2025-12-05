import torch
import numpy as np
import cv2
import os

DAYLIGHT_OPENINGS_LABEL_KEY = 4
WINDOW_MASK_FOLDER = "Masks/WindowMasks/"
SIDE_MIRROR_MASK_FOLDER = "Masks/SideMirrorMasks/"
ROTATED_LABELS_FOLDER = "HorizontalAlign/RotatedLabels/"
OUTPUT_FOLDER = "Masks/ProcessedMasks/"


def rotatedLabelsToDict(rotatedLabelText):
    lines = rotatedLabelText.strip().split('\n')
    output = {}
    for line in lines[1:]:  # Skip the first line (angle)
        parts = line.split(' ')
        key = int(parts[0])
        coords = list(map(float, parts[1:]))
        points = []
        for i in range(0, len(coords), 2):
            points.append((coords[i], coords[i+1]))
        output.setdefault(key, []).append(points)
    return output


def mergeWindowAndMirror(window_mask, mirror_mask, max_gap=20):
    """
    Merge window and mirror masks by filling small gaps between them.
    Only fills the gap region between the two masks, without expanding them in other directions.
    
    Args:
        window_mask: Boolean tensor of the window mask
        mirror_mask: Boolean tensor of the mirror mask
        max_gap: Maximum gap size (in pixels) to fill between masks
    
    Returns:
        Combined mask with gaps filled
    """
    window_np = window_mask.cpu().numpy().astype(np.uint8)
    mirror_np = mirror_mask.cpu().numpy().astype(np.uint8)
    
    # Create dilation kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_gap, max_gap))
    
    # Dilate both masks
    window_dilated = cv2.dilate(window_np, kernel, iterations=1)
    mirror_dilated = cv2.dilate(mirror_np, kernel, iterations=1)
    
    # Find the overlap region between dilated masks - this is the gap region
    gap_region = (window_dilated > 0) & (mirror_dilated > 0)
    
    # Remove pixels that are already part of either mask
    gap_region = gap_region & ~(window_np > 0) & ~(mirror_np > 0)
    
    # Combine original masks with the gap region
    combined = (window_np > 0) | (mirror_np > 0) | gap_region
    
    return torch.from_numpy(combined).bool().to(window_mask.device)


def smoothMask(mask, mirror_mask, kernel_size=5, iterations=3, frontWindowIterations=3):
    """
    Smooth the mask edges to remove randomness and jagged boundaries.
    Applies smoothing to each connected component separately, then combines them.
    The front window and mirror combination receives more smoothing iterations.
    
    Args:
        mask: Boolean tensor of the combined mask to smooth
        mirror_mask: Boolean tensor of the mirror mask (used to identify front window + mirror component)
        kernel_size: Size of the morphological kernel (larger = more smoothing)
        iterations: Number of times to apply smoothing to non-front-window masks
        frontWindowIterations: Number of times to apply smoothing to front window + mirror
    
    Returns:
        Smoothed mask as a boolean tensor
    """
    mask_np = mask.cpu().numpy().astype(np.uint8)
    mirror_np = mirror_mask.cpu().numpy().astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask_np)
    
    # Identify which component contains the mirror (front window + mirror combination)
    front_window_label = None
    for label_id in range(1, num_labels):
        component_mask = (labels == label_id)
        # Check if this component overlaps with the mirror mask
        if np.any(component_mask & (mirror_np > 0)):
            front_window_label = label_id
            break
    
    # Create output mask
    smoothed_combined = np.zeros_like(mask_np)
    
    # Create a circular kernel for smoother results
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Smooth each connected component separately
    for label_id in range(1, num_labels):  # Skip background (label 0)
        component_mask = (labels == label_id).astype(np.uint8)
        
        # Use more iterations for front window + mirror combination
        iters = frontWindowIterations if label_id == front_window_label else iterations
        
        # Opening (erosion followed by dilation) removes small protrusions
        smoothed = cv2.morphologyEx(component_mask, cv2.MORPH_OPEN, kernel, iterations=iters)
        
        # Closing (dilation followed by erosion) fills small holes and gaps
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel, iterations=iters)
        
        # Apply Gaussian blur and threshold for additional smoothing
        smoothed_float = smoothed.astype(np.float32)
        smoothed_float = cv2.GaussianBlur(smoothed_float, (kernel_size, kernel_size), 0)
        smoothed = (smoothed_float > 0.5).astype(np.uint8)
        
        # Add to combined result
        smoothed_combined = smoothed_combined | smoothed
    
    return torch.from_numpy(smoothed_combined).bool().to(mask.device)


def preprocessMasks(baseFileName):
    window_masks = torch.load(os.path.join(WINDOW_MASK_FOLDER, baseFileName + ".pt"))
    mirror_masks = torch.load(os.path.join(SIDE_MIRROR_MASK_FOLDER, baseFileName + ".pt"))
    
    # Combine individual masks into single window and mirror masks
    window_mask = (window_masks.sum(dim=0) > 0)[0]
    mirror_mask = (mirror_masks.sum(dim=0) > 0)[0]

    # Only consider masks within the label for daylight openings
    rotatedLabelsPath = os.path.join(ROTATED_LABELS_FOLDER, baseFileName + ".txt")
    with open(rotatedLabelsPath, 'r') as file:
        rotatedLabelsText = file.read()
    labelDict = rotatedLabelsToDict(rotatedLabelsText)
    assert DAYLIGHT_OPENINGS_LABEL_KEY in labelDict, "No daylight openings in label"
    assert len(labelDict[DAYLIGHT_OPENINGS_LABEL_KEY]) == 1, f"Expected exactly one daylight opening, found {len(labelDict[DAYLIGHT_OPENINGS_LABEL_KEY])}"
    dloPolygon = labelDict[DAYLIGHT_OPENINGS_LABEL_KEY][0]
    assert len(dloPolygon) == 4, f"Daylight opening polygon should have 4 corners, found {len(dloPolygon)}"
    height, width = window_mask.shape
    scaled_polygon = np.array([(x * width, y * height) for x, y in dloPolygon], dtype=np.int32)
    dlo_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(dlo_mask, [scaled_polygon], 1)
    dlo_mask_tensor = torch.from_numpy(dlo_mask).bool().to(window_mask.device)
    
    # Apply daylight opening mask to both window and mirror masks
    window_mask = window_mask & dlo_mask_tensor
    mirror_mask = mirror_mask & dlo_mask_tensor
    
    # Merge window and mirror masks, filling any small gaps between them
    masks = mergeWindowAndMirror(window_mask, mirror_mask)
    
    # Smooth the combined mask to remove jagged edges
    masks = smoothMask(masks, mirror_mask)
    
    return masks


def run():
    mask_files = [f for f in os.listdir(WINDOW_MASK_FOLDER) if f.endswith('.pt')]
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for mask_file in mask_files:
        baseFileName = mask_file.split('.')[0]
        try:
            masks = preprocessMasks(baseFileName)
            # Save the processed mask with shape (1, 1, H, W)
            reshapedMasks = torch.reshape(masks, (1, 1, *masks.shape))
            torch.save(reshapedMasks, os.path.join(OUTPUT_FOLDER, f"{baseFileName}.pt"))
            print(f"Processed {baseFileName}")
        except Exception as e:
            print(f"Error processing {mask_file}: {e}")


if __name__ == "__main__":
    run()