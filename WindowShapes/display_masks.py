import os
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from image_files import list_image_files

from config import (
    HORIZONTAL_ALIGN_IMAGES as IMAGE_FOLDER,
    PROCESSED_MASK_FOLDER as WINDOW_MASK_FOLDER,
    SIDE_MIRROR_MASK_FOLDER as MIRROR_MASK_FOLDER,
)


def combine_masks(masks):
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


def overlay_boolean_mask(mask_overlay: Image.Image, boolean_mask: np.ndarray, color) -> Image.Image:
    if boolean_mask is None or not boolean_mask.any():
        return mask_overlay
    mask_uint8 = (boolean_mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_uint8, mode="L")
    single_mask_img = Image.new("RGBA", mask_overlay.size, (0, 0, 0, 0))
    single_mask_img.paste(color, (0, 0), mask=mask_pil)
    return Image.alpha_composite(mask_overlay, single_mask_img)


def visualize_masks_on_image(image_name: str) -> None:
    # Load an image
    image = Image.open(os.path.join(IMAGE_FOLDER, image_name))
    # Convert the image to RGBA to support transparency
    image = image.convert("RGBA")

    # Load window masks
    base_name = os.path.splitext(image_name)[0]
    window_mask_path = os.path.join(WINDOW_MASK_FOLDER, f"{base_name}.pt")
    window_combined = None
    if os.path.exists(window_mask_path):
        window_masks = torch.load(window_mask_path, weights_only=False)
        window_combined = combine_masks(window_masks)

    # Load mirror masks
    mirror_combined = None
    mirror_mask_path = os.path.join(MIRROR_MASK_FOLDER, f"{base_name}.pt")
    if os.path.exists(mirror_mask_path):
        mirror_masks = torch.load(mirror_mask_path, weights_only=False)
        mirror_combined = combine_masks(mirror_masks)

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

    # Add regions
    mask_overlay = overlay_boolean_mask(mask_overlay, window_only, (255, 0, 0, 128))  # red
    mask_overlay = overlay_boolean_mask(mask_overlay, mirror_only, (0, 255, 0, 128))  # green
    mask_overlay = overlay_boolean_mask(mask_overlay, overlap, (0, 0, 255, 128))  # blue

    # Combine the original image with the mask overlay
    result_image = Image.alpha_composite(image, mask_overlay)

    # Display the result
    plt.imshow(result_image)
    plt.axis('off')
    plt.show()


# Backwards-compatible aliases (deprecated)
combineMasks = combine_masks
visualizeMasksOnImage = visualize_masks_on_image


if __name__ == "__main__":
    files = list_image_files(IMAGE_FOLDER)
    shuffle(files)
    for file in files:
        if os.path.exists(os.path.join(WINDOW_MASK_FOLDER, f"{file.split('.')[0]}.pt")):
            visualize_masks_on_image(file)