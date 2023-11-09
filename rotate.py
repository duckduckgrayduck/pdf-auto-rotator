import argparse
import fitz
import cv2
import numpy as np
from fitz import Pixmap

def skew_correction(image):
    # Apply edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Detect lines in the image using Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Calculate skew angle
    skew_angle = 0
    if lines is not None and len(lines) > 0:
        total_skew_angle = 0
        total_lines = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            total_skew_angle += angle
        skew_angle = total_skew_angle / total_lines
    
    # Correct skew in the image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    corrected_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return corrected_image, skew_angle

def main(input_pdf, output_pdf):
    pdf_document = fitz.open(input_pdf)

    # Create a new PDF document for the corrected images
    output_pdf_document = fitz.open()

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]

        # Determine page orientation (portrait or landscape)
        page_width = int(page.rect.width)
        page_height = int(page.rect.height)
        is_landscape = page_width > page_height

        images = page.get_images(full=True)

        for img_index, image in enumerate(images):
            base_image = pdf_document.extract_image(image[0])
            image_data = base_image["image"]

            # Convert image data to NumPy array
            image_np = np.frombuffer(image_data, dtype=np.uint8)

            # Decode image using OpenCV (assuming BGRA format)
            image_cv = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)

            # Correct skew in the image using the skew_correction method
            corrected_image, skew_angle = skew_correction(image_cv)

            # Calculate the correct coordinates for the image
            x0, y0, x1, y1 = int(image[0]), int(image[1]), int(image[2]), int(image[3])
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(page_width, x1), min(page_height, y1)

            # Calculate offset to center the image within the page
            image_width = x1 - x0
            image_height = y1 - y0
            if is_landscape:
                # For landscape pages, swap width and height before calculating offsets
                page_width, page_height = page_height, page_width
            x_offset = (page_width - image_width) // 2
            y_offset = (page_height - image_height) // 2

            # Create a Rect object with the corrected coordinates and offset
            existing_image_rect = fitz.Rect(x0 + x_offset, y0 + y_offset, x1 + x_offset, y1 + y_offset)

            # Convert corrected image to image buffer
            image_buffer = cv2.imencode('.png', corrected_image)[1].tobytes()

            # Convert image buffer to Pixmap object
            corrected_pixmap = Pixmap(image_buffer)

            # Insert the corrected image into the new PDF document with centered alignment in portrait or landscape orientation
            output_pdf_document.new_page(width=page_width, height=page_height)
            output_pdf_document[-1].insert_image(existing_image_rect, pixmap=corrected_pixmap)

    # Save the new PDF document with corrected images
    output_pdf_document.save(output_pdf)
    output_pdf_document.close()

    # Close the original PDF document
    pdf_document.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correct skew in images within a PDF file.")
    parser.add_argument("input_pdf", help="Input PDF file path")
    parser.add_argument("output_pdf", help="Output PDF file path")
    args = parser.parse_args()
    main(args.input_pdf, args.output_pdf)
