import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ORIGINAL_IMAGE_FOLDER = os.path.join(BASE_DIR, "CleanedImages")
ORIGINAL_LABELS_FOLDER = os.path.join(BASE_DIR, "OriginalLabels")

MASK_DIR = os.path.join(BASE_DIR, "Masks")
WINDOW_MASK_FOLDER = os.path.join(MASK_DIR, "WindowMasks")
SIDE_MIRROR_MASK_FOLDER = os.path.join(MASK_DIR, "SideMirrorMasks")
PROCESSED_MASK_FOLDER = os.path.join(MASK_DIR, "ProcessedMasks")

POLYGON_FOLDER = os.path.join(BASE_DIR, "Polygons")

# Horizontal alignment folders
HORIZONTAL_ALIGN_DIR = os.path.join(BASE_DIR, "HorizontalAlign")
HORIZONTAL_ALIGN_IMAGES = os.path.join(HORIZONTAL_ALIGN_DIR, "Images")
ROTATED_LABELS_FOLDER = os.path.join(HORIZONTAL_ALIGN_DIR, "RotatedLabels")

DXF_FOLDER = os.path.join(BASE_DIR, "DXF")
SCALED_DXF_FOLDER = os.path.join(BASE_DIR, "ScaledDXF")

POLYGON_CORNERS = 10