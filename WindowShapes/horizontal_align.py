"""Horizontally align car images using tire labels.

This script rotates each image so that the bottoms of the labeled tires are aligned.
Images are skipped if the tire labels are missing/invalid.
"""

import math
import os

from PIL import Image

from image_files import find_image_path, list_image_files, split_base_and_ext


TIRE_LABEL_KEY = 0  # The key for tire locations in the label files

from config import (
    ORIGINAL_IMAGE_FOLDER as IMAGE_FOLDER,
    ORIGINAL_LABELS_FOLDER as LABEL_FOLDER,
    HORIZONTAL_ALIGN_IMAGES as OUTPUT_IMAGE_FOLDER,
    ROTATED_LABELS_FOLDER as OUTPUT_LABEL_FOLDER
)

def label_text_to_dict(label_text: str) -> dict[int, list[tuple[float, float, float, float]]]:
    output = {}
    lines = label_text.strip().split("\n")
    for line in lines:
        parts = line.split(" ")
        assert len(parts) == 5, f"Invalid label format: {line}"
        k = int(parts[0])
        rect = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        # The rect format is: (centerX, centerY, width, height)
        output.setdefault(k, []).append(rect)
    return output

def get_rotation_angle(label_dict, tire_label_key: int) -> float:
    assert tire_label_key in label_dict, f"Tire label key {tire_label_key} not found in label dictionary"
    assert len(label_dict[tire_label_key]) == 2, f"Expected 2 tires, found {len(label_dict[tire_label_key])}"

    tires = label_dict[tire_label_key]
    tires.sort(key=lambda rect: rect[0])  # Sort by x-coordinate
    left_tire_x = tires[0][0]
    right_tire_x = tires[1][0]
    left_tire_bottom_y = tires[0][1] + tires[0][3] / 2
    right_tire_bottom_y = tires[1][1] + tires[1][3] / 2
    angle = math.atan2((right_tire_bottom_y - left_tire_bottom_y), (right_tire_x - left_tire_x))
    # math.atan2(y,x) returns arctan(y/x)
    return angle / math.pi * 180

def rotate_image(image: Image.Image, angle_degrees: float) -> Image.Image:
    return image.rotate(angle_degrees)

def rotate_label(label_dict, angle_degrees: float) -> str:
    output = [str(angle_degrees)]
    angle_rad = math.radians(angle_degrees)
    cos = math.cos(angle_rad)
    sin = math.sin(angle_rad)

    for key, rects in label_dict.items():
        for rect in rects:
            cx, cy, w, h = rect
            corners = [
                (cx - w/2, cy - h/2),
                (cx + w/2, cy - h/2),
                (cx + w/2, cy + h/2),
                (cx - w/2, cy + h/2)
            ]
            rotatedCoords = []
            for x, y in corners:
                tx = x - 0.5
                ty = y - 0.5
                rx = tx * cos + ty * sin
                ry = -tx * sin + ty * cos
                final_x = rx + 0.5
                final_y = ry + 0.5
                rotatedCoords.append(final_x)
                rotatedCoords.append(final_y)

            line = f"{key} {' '.join(map(str, rotatedCoords))}"
            output.append(line)
            
    return "\n".join(output)

def align_image(name: str, image_folder: str, label_folder: str, tire_label_key: int) -> None:
    # Support .jpg/.jpeg/.png input images
    image_path = find_image_path(image_folder, name)
    label_path = os.path.join(label_folder, f"{name}.txt")
    if not image_path or not os.path.exists(image_path) or not os.path.exists(label_path):
        print(f"Missing image or label for {name}")
        return

    image = Image.open(image_path)
    with open(label_path, "r", encoding="utf-8") as file:
        label_text = file.read()

    label_dict = label_text_to_dict(label_text)

    angle = get_rotation_angle(label_dict, tire_label_key)
    if not os.path.exists(OUTPUT_IMAGE_FOLDER):
        os.makedirs(OUTPUT_IMAGE_FOLDER)
    if not os.path.exists(OUTPUT_LABEL_FOLDER):
        os.makedirs(OUTPUT_LABEL_FOLDER)
    rotated_image = rotate_image(image, angle)
    # Preserve original extension when saving aligned image
    _, ext = os.path.splitext(image_path)
    rotated_image.save(os.path.join(OUTPUT_IMAGE_FOLDER, f"{name}{ext.lower()}"))
    rotated_label_content = rotate_label(label_dict, angle)
    with open(os.path.join(OUTPUT_LABEL_FOLDER, f"{name}.txt"), "w", encoding="utf-8") as file:
        file.write(rotated_label_content)


# Backwards-compatible aliases (deprecated)
labelTextToDict = label_text_to_dict
getRotationAngle = get_rotation_angle
rotateImage = rotate_image
rotateLabel = rotate_label
alignImage = align_image


def run():
    for file in list_image_files(IMAGE_FOLDER):
        name, _ = split_base_and_ext(file)
        try:
            align_image(name, IMAGE_FOLDER, LABEL_FOLDER, TIRE_LABEL_KEY)
        except AssertionError as e:
            print(f"Skipping {name} due to error: {e}")

if __name__ == "__main__":
    run()