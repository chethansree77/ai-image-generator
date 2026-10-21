from datetime import datetime
import os


def create_image_filename():
    """
    Create a unique filename for the generated image.
    """

    # Make sure generated_images folder exists
    os.makedirs("generated_images", exist_ok=True)

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Return the complete file path
    return f"generated_images/image_{timestamp}.png"