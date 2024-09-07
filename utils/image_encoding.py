import base64

def encode_image(image_path: str) -> str:
    """Helper function to encode an image file as base64.
    Implementation taken from here: https://platform.openai.com/docs/guides/vision
    
    Args:
        image_path (str): Path to the image file.

    Returns:
        str: The image file encoded as base64.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')