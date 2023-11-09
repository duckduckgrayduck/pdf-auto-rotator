import fitz
import cv2
import numpy as np
from fitz import Pixmap

def skew_correction(image):
    # Check the number of channels in the input image
    if image.shape[-1] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.shape[-1] == 4: 
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        # If the number of channels is not 3 or 4, handle it accordingly (e.g., raise an error or convert to grayscale)
        raise ValueError("Unsupported number of channels in the input image")

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines in the image using Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    # Calculate skew angle
    skew_angle = 0
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            if theta != 0:
                skew_angle += theta
        skew_angle /= len(lines)
        skew_angle = np.degrees(skew_angle) - 90  # Convert radians to degrees and adjust for vertical text

    # Rotate the image to correct skew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    corrected_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return corrected_image, skew_angle


input_pdf_path = "input.pdf"
output_pdf_path = "output.pdf"

pdf_document = fitz.open(input_pdf_path)

# Create a new PDF document for the corrected images
output_pdf_document = fitz.open()

for page_num in range(pdf_document.page_count):
    page = pdf_document[page_num]
    portrait_width = int(min(page.rect.width, page.rect.height))
    portrait_height = int(max(page.rect.width, page.rect.height))
    
    images = page.get_images(full=True)
    
    for img_index, image in enumerate(images):
        base_image = pdf_document.extract_image(image[0])
        image_data = base_image["image"]

        # Convert image data to NumPy array
        image_np = np.frombuffer(image_data, dtype=np.uint8)

        # Decode image using OpenCV
        image_cv = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)

        # Correct skew in the image
        corrected_image, skew_angle = skew_correction(image_cv)

        # Calculate the correct coordinates for the image
        x0, y0, x1, y1 = int(image[0]), int(image[1]), int(image[2]), int(image[3])
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(portrait_width, x1), min(portrait_height, y1)

        # Calculate offset to center the image within the portrait-oriented page
        image_width = x1 - x0
        image_height = y1 - y0
        x_offset = (portrait_width - image_width) // 2
        y_offset = (portrait_height - image_height) // 2

        # Create a Rect object with the corrected coordinates and offset
        existing_image_rect = fitz.Rect(x0 + x_offset, y0 + y_offset, x1 + x_offset, y1 + y_offset)

        # Apply skew angle correction to the Pixmap object
        rotation_matrix = cv2.getRotationMatrix2D((corrected_image.shape[1] // 2, corrected_image.shape[0] // 2), skew_angle, 1.0)
        corrected_image = cv2.warpAffine(corrected_image, rotation_matrix, (corrected_image.shape[1], corrected_image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Convert corrected image to image buffer
        image_buffer = cv2.imencode('.png', corrected_image)[1].tobytes()

        # Convert image buffer to Pixmap object
        corrected_pixmap = Pixmap(image_buffer)

        # Insert the corrected image into the new PDF document with centered alignment in portrait orientation
        output_pdf_document.new_page(width=portrait_width, height=portrait_height)  # Use portrait dimensions
        output_pdf_document[-1].insert_image(existing_image_rect, pixmap=corrected_pixmap)

# Save the new PDF document with corrected images
output_pdf_document.save(output_pdf_path)
output_pdf_document.close()
pdf_document.close()