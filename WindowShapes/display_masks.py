import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from random import shuffle
from image_files import list_image_files

from config import (
    HORIZONTAL_ALIGN_IMAGES as IMAGE_FOLDER,
    PROCESSED_MASK_FOLDER as WINDOW_MASK_FOLDER
    #WINDOW_MASK_FOLDER,
    #SIDE_MIRROR_MASK_FOLDER as MIRROR_MASK_FOLDER
)
MIRROR_MASK_FOLDER = None


def addMasksToOverlay(masks, mask_overlay, image_size, color):
    """Add masks to an overlay image with the specified color."""
    for i, mask in enumerate(masks):
        # Convert mask to numpy array if it's a tensor
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Remove extra dimensions
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        # Ensure mask is boolean
        mask = mask > 0
        
        # Create a temporary image for this single mask
        single_mask_img = Image.new("RGBA", image_size, (0, 0, 0, 0))
        
        # Convert the boolean mask to an image mask
        # Where mask is True, apply the color
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode='L')
        
        # Paste the color using the mask
        single_mask_img.paste(color, (0, 0), mask=mask_pil)
        
        # Composite this mask onto the main overlay
        mask_overlay = Image.alpha_composite(mask_overlay, single_mask_img)
    
    return mask_overlay


def combineMasks(masks):
    """Combine multiple masks into a single boolean mask."""
    combined = None
    for mask in masks:
        # Convert mask to numpy array if it's a tensor
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Remove extra dimensions
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        
        # Ensure mask is boolean
        mask = mask > 0
        
        if combined is None:
            combined = mask
        else:
            combined = combined | mask
    
    return combined


def visualizeMasksOnImage(imageName):
    # Load an image
    image = Image.open(os.path.join(IMAGE_FOLDER, imageName))
    # Convert the image to RGBA to support transparency
    image = image.convert("RGBA")

    # Load window masks
    window_mask_path = os.path.join(WINDOW_MASK_FOLDER, f"{imageName.split('.')[0]}.pt")
    window_combined = None
    if os.path.exists(window_mask_path):
        window_masks = torch.load(window_mask_path)
        window_combined = combineMasks(window_masks)

    # Load mirror masks
    mirror_combined = None
    if MIRROR_MASK_FOLDER is not None:
        mirror_mask_path = os.path.join(MIRROR_MASK_FOLDER, f"{imageName.split('.')[0]}.pt")
        if os.path.exists(mirror_mask_path):
            mirror_masks = torch.load(mirror_mask_path)
            mirror_combined = combineMasks(mirror_masks)

    # Create a blank image for the masks with the same size as the original image
    mask_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

    # Calculate overlap and exclusive regions
    if window_combined is not None and mirror_combined is not None:
        overlap = window_combined & mirror_combined
        window_only = window_combined & ~mirror_combined
        mirror_only = mirror_combined & ~window_combined
    elif window_combined is not None:
        overlap = None
        window_only = window_combined
        mirror_only = None
    elif mirror_combined is not None:
        overlap = None
        window_only = None
        mirror_only = mirror_combined
    else:
        overlap = None
        window_only = None
        mirror_only = None

    # Add window-only regions (red)
    if window_only is not None and window_only.any():
        window_color = (255, 0, 0, 128)  # Red with 50% opacity
        mask_uint8 = (window_only * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode='L')
        single_mask_img = Image.new("RGBA", image.size, (0, 0, 0, 0))
        single_mask_img.paste(window_color, (0, 0), mask=mask_pil)
        mask_overlay = Image.alpha_composite(mask_overlay, single_mask_img)

    # Add mirror-only regions (green)
    if mirror_only is not None and mirror_only.any():
        mirror_color = (0, 255, 0, 128)  # Green with 50% opacity
        mask_uint8 = (mirror_only * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode='L')
        single_mask_img = Image.new("RGBA", image.size, (0, 0, 0, 0))
        single_mask_img.paste(mirror_color, (0, 0), mask=mask_pil)
        mask_overlay = Image.alpha_composite(mask_overlay, single_mask_img)

    # Add overlap regions (blue)
    if overlap is not None and overlap.any():
        overlap_color = (0, 0, 255, 128)  # Blue with 50% opacity
        mask_uint8 = (overlap * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode='L')
        single_mask_img = Image.new("RGBA", image.size, (0, 0, 0, 0))
        single_mask_img.paste(overlap_color, (0, 0), mask=mask_pil)
        mask_overlay = Image.alpha_composite(mask_overlay, single_mask_img)

    # Combine the original image with the mask overlay
    result_image = Image.alpha_composite(image, mask_overlay)

    # Display the result
    plt.imshow(result_image)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    files = list_image_files(IMAGE_FOLDER)
    shuffle(files)
    for file in files:
        if os.path.exists(os.path.join(WINDOW_MASK_FOLDER, f"{file.split('.')[0]}.pt")):
            visualizeMasksOnImage(file)