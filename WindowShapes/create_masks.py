import os
import sys

import torch
from PIL import Image

# Ensure the local `sam3` package is importable when running this script directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam3"))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from image_files import is_image_file


def create_mask_from_image(image_file_path: str, processor: Sam3Processor, prompt: str):
    image = Image.open(image_file_path)
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    return output["masks"]


def create_masks(processor: Sam3Processor, image_folder: str, output_folder: str, prompt: str) -> None:
    os.makedirs(output_folder, exist_ok=True)

    for image_file_name in os.listdir(image_folder):
        if not is_image_file(image_file_name):
            continue
        try:
            image_path = os.path.join(image_folder, image_file_name)
            mask = create_mask_from_image(image_path, processor, prompt)
            base_name = os.path.splitext(image_file_name)[0]
            mask_file_name = os.path.join(output_folder, f"{base_name}.pt")
            torch.save(mask, mask_file_name)
        except Exception as e:
            print(f"Error processing {image_file_name}: {e}")


def run(image_folder: str, output_folder: str, prompt: str) -> None:
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    create_masks(processor, image_folder, output_folder, prompt)


# Backwards-compatible aliases (deprecated)
createMaskFromImage = create_mask_from_image
createMasks = create_masks


if __name__ == "__main__":
    from config import HORIZONTAL_ALIGN_IMAGES, SIDE_MIRROR_MASK_FOLDER, WINDOW_MASK_FOLDER

    print("Creating window masks...")
    run(HORIZONTAL_ALIGN_IMAGES, WINDOW_MASK_FOLDER, "Side window")

    print("Creating side mirror masks...")
    run(HORIZONTAL_ALIGN_IMAGES, SIDE_MIRROR_MASK_FOLDER, "Side mirror")