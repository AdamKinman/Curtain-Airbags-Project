# Curtain-Airbags-Project

## How to use

### Generating window shape labels

1. RECOMMENDED: initialize and activate a python virtual environment (venv) in the root project folder.

2. Install the required python libraries (pip install -r WindowShapes/requirements.txt).

3. Clone the sam3 github repository (https://github.com/facebookresearch/sam3) and place it inside the WindowShapes folder (so that the root directory of the sam3 repository is WindowShapes/sam3). Install the sam3 model weights according to the instructions found in sam3/README.md.

4. Create a file WindowShapes/.env, and add an API key to the Google Gemini API as in WindowShapes/.env_example. This step is only required for automatically re-scaling generated DXF-files. All other functionality of the project can be used without an API key.

5. Run WindowShapes/main.py to open a graphical user interface where the following can be done:

- Horizontally align images of cars. (This is only possible for images in the GP22 dataset, since it requires the original dataset labels.)
- Generate pixel-masks for the side window of the cars using sam3.
- Apply post-processing to the masks. This is mainly for expanding the masks with an approximation of the window area behind the side mirror (since this area does not get included in the sam3 masks).
- Generate polygons describing the outlines of sam3-masks.
- Generate DXF-files based on the polygons.
- Scale the DXF-files to reflect the real-life scale of the car windows. (Uses the Gemini API).
- Generate new side-view images of cars using Stable Diffusion.
- Browse and display files (images, masks, polygons or DXF).