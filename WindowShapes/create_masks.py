import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sam3'))

import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def createMaskFromImage(imageFileName, processor, PROMPT):
    image = Image.open(imageFileName)
    interference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=interference_state, prompt=PROMPT)
    return output["masks"]

def createMasks(processor, IMAGE_FOLDER, OUTPUT_FOLDER, PROMPT):
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    for imageFileName in os.listdir(IMAGE_FOLDER):
        if not imageFileName.lower().endswith('.jpg'): continue
        try:
            mask = createMaskFromImage(os.path.join(IMAGE_FOLDER, imageFileName), processor, PROMPT)
            maskFileName = os.path.join(OUTPUT_FOLDER, f"{imageFileName.split('.')[0]}.pt")
            torch.save(mask, maskFileName)
        except Exception as e:
            print(f"Error processing {imageFileName}: {e}")


def run(IMAGE_FOLDER, OUTPUT_FOLDER, PROMPT):
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    createMasks(processor, IMAGE_FOLDER, OUTPUT_FOLDER, PROMPT)


if __name__ == '__main__':
    from config import (HORIZONTAL_ALIGN_IMAGES, WINDOW_MASK_FOLDER, SIDE_MIRROR_MASK_FOLDER)
    print("Creating window masks...")
    run(
        HORIZONTAL_ALIGN_IMAGES,
        WINDOW_MASK_FOLDER,
        "Side window"
    )
    print("Creating side mirror masks...")
    run(
        HORIZONTAL_ALIGN_IMAGES,
        SIDE_MIRROR_MASK_FOLDER,
        "Side mirror"
    )