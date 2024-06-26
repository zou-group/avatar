import numpy as np
from PIL import Image


def extract_patch(image: Image.Image, box: tuple) -> Image.Image:
    """
    Extract a patch from the image based on the given bounding box.

    Args:
        image (Image.Image): The input image from which the patch will be extracted.
        box (tuple): The bounding box coordinates (x1, y1, x2, y2).

    Returns:
        Image.Image: The extracted patch as an Image object.
    """
    x1, y1, x2, y2 = box
    return image.crop((x1, y1, x2, y2))
