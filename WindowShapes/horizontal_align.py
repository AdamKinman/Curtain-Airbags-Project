from PIL import Image
import os
import math

'''
This scripts rotates all images so that the bottoms of the tires are aligned horizontally.
Images will be skipped if the tires are already perfectly aligned or if there is an issue
with the associated label file, usually that more than two tires are labeled.
'''

TIRE_LABEL_KEY = 0 # The key for tire locations in the label files

from config import (
    ORIGINAL_IMAGE_FOLDER as IMAGE_FOLDER,
    ORIGINAL_LABELS_FOLDER as LABEL_FOLDER,
    HORIZONTAL_ALIGN_IMAGES as OUTPUT_IMAGE_FOLDER,
    ROTATED_LABELS_FOLDER as OUTPUT_LABEL_FOLDER
)

'''
Output labels format:
Angle (degrees)
key x1 y1 x2 y2 x3 y3 x4 y4 (key is the same as in the original label, 
the rest are the four corners of the rotated bounding box)
'''

def labelTextToDict(labelText):
    output = {}
    lines = labelText.strip().split('\n')
    for line in lines:
        parts = line.split(' ')
        assert len(parts) == 5, f"Invalid label format: {line}"
        k = int(parts[0])
        rect = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        # The rect format is: (centerX, centerY, width, height)
        output.setdefault(k, []).append(rect)
    return output

def getRotationAngle(labelDict, tireLabelKey):
    assert tireLabelKey in labelDict, f"Tire label key {tireLabelKey} not found in label dictionary"
    assert len(labelDict[tireLabelKey]) == 2, f"Expected 2 tires, found {len(labelDict[tireLabelKey])}"
    tires = labelDict[tireLabelKey]
    tires.sort(key=lambda rect: rect[0])  # Sort by x-coordinate
    leftTireX = tires[0][0]
    rightTireX = tires[1][0]
    leftTireBottomY = tires[0][1] + tires[0][3]/2
    rightTireBottomY = tires[1][1] + tires[1][3]/2
    angle = math.atan2((rightTireBottomY - leftTireBottomY), (rightTireX - leftTireX))
    # math.atan2(y,x) returns arctan(y/x)
    return angle / math.pi * 180

def rotateImage(image, angle):
    rotatedImage = image.rotate(angle)
    return rotatedImage

def rotateLabel(labelDict, angle, imageSize):
    output = [str(angle)]
    angleRad = math.radians(angle)
    cos = math.cos(angleRad)
    sin = math.sin(angleRad)

    for key, rects in labelDict.items():
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

def alignImage(name, imageFolder, labelFolder, tireLabelKey):
    imagePath = os.path.join(imageFolder, f"{name}.jpg")
    labelPath = os.path.join(labelFolder, f"{name}.txt")
    if not os.path.exists(imagePath) or not os.path.exists(labelPath):
        print(f"Missing image or label for {name}")
        return

    image = Image.open(imagePath)
    with open(labelPath, 'r') as file:
        labelText = file.read()
    labelDict = labelTextToDict(labelText)

    angle = getRotationAngle(labelDict, tireLabelKey)
    if not os.path.exists(OUTPUT_IMAGE_FOLDER):
        os.makedirs(OUTPUT_IMAGE_FOLDER)
    if not os.path.exists(OUTPUT_LABEL_FOLDER):
        os.makedirs(OUTPUT_LABEL_FOLDER)
    rotatedImage = rotateImage(image, angle)
    rotatedImage.save(os.path.join(OUTPUT_IMAGE_FOLDER, f"{name}.jpg"))
    rotatedLabelContent = rotateLabel(labelDict, angle, image.size)
    with open(os.path.join(OUTPUT_LABEL_FOLDER, f"{name}.txt"), 'w') as file:
        file.write(rotatedLabelContent)


def run():
    files = os.listdir(IMAGE_FOLDER)
    for file in files:
        if file.endswith('.jpg'):
            name = os.path.splitext(file)[0]
            try:
                alignImage(name, IMAGE_FOLDER, LABEL_FOLDER, TIRE_LABEL_KEY)
            except AssertionError as e:
                print(f"Skipping {name} due to error: {e}")

if __name__ == '__main__':
    run()