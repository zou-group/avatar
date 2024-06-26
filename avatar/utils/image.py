
import base64
from PIL import Image
from io import BytesIO


def image_to_base64(pil_img, img_format='JPEG'):
    """
    Convert a PIL Image object to a base64 encoded string.

    Args:
    pil_img (PIL.Image.Image): The PIL Image object to convert.
    img_format (str): The format to save the image as (e.g., 'PNG', 'JPEG').

    Returns:
    str: Base64 encoded string of the image.
    """
    # Create a BytesIO object to hold the image data
    buffered = BytesIO()
    
    # Save the image to the buffer in the specified format
    pil_img.save(buffered, format=img_format)
    
    # Retrieve the byte data from the buffer
    img_byte = buffered.getvalue()
    
    # Encode the byte data to base64
    base64_image = base64.b64encode(img_byte).decode('utf-8')
    
    return base64_image
