from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import cv2
from typing import Literal
from loguru import logger
from scipy.signal import convolve2d

class ImageProcessor:
    _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="vi")
    
    @staticmethod
    def enhance_image(image: Image.Image) -> Image.Image:
        """Enhance the image to improve OCR accuracy while preserving fine details."""

        # Convert to numpy array
        image_np = np.array(image)

        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Apply Bilateral Filtering (denoise while keeping edges)
        filtered = cv2.bilateralFilter(gray, d=7, sigmaColor=30, sigmaSpace=30)

        # Apply a mild sharpening kernel
        sharpening_kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        sharpened = cv2.filter2D(filtered, -1, sharpening_kernel)

        # Use Adaptive Histogram Equalization (CLAHE) for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(sharpened)

        logger.info("Image enhanced (clean, sharp, and balanced).")

        return Image.fromarray(contrast_enhanced)


    @staticmethod
    def detect_rotation_contour(image: Image.Image) -> float:
        """Detects the skew angle of text using contour-based rotation."""
        # Convert PIL image to grayscale NumPy array
        image_np = np.array(image.convert("L"))

        # Apply binary thresholding
        _, thresh = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning("No text detected.")
            return 0

        # Get the minimum area bounding box for the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[-1]

        # Adjust angle to be within -45 to +45 degrees
        if angle < -45:
            angle += 90

        logger.info(f"Detected rotation angle: {angle}°")
        return angle

    @staticmethod
    def detect_rotation_hough(image: Image.Image) -> int:
        """Detects rotation using Hough Line Transform and returns 90° or 180° correction."""
        # Convert PIL image to grayscale NumPy array
        image_np = np.array(image.convert("L"))

        # Apply edge detection
        edges = cv2.Canny(image_np, 50, 150, apertureSize=3)

        # Detect lines using Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is None:
            logger.warning("No lines detected.")
            return 0  # No rotation needed

        # Compute the dominant angle
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90  # Convert from radians to degrees
            angles.append(angle)

        avg_angle = np.median(angles)  # Use median to avoid outliers

        # Normalize to closest 90° or 180°
        if -135 < avg_angle < -45:
            corrected_angle = 90
        elif -180 < avg_angle < -135 or 45 < avg_angle < 135:
            corrected_angle = 180
        else:
            corrected_angle = 0  # No rotation needed

        logger.info(f"Detected skew: {avg_angle}° -> Rotating by {corrected_angle}°")
        return corrected_angle

    @staticmethod
    def detect_rotation_paddle(image: Image.Image) -> float | Literal[0]:
        """Detect the rotation angle of an image using PaddleOCR."""

        # Preprocess image
        image = ImageProcessor.preprocess_image(image)

        # Ensure image is in RGB mode
        if image.mode in ("RGBA", "L"):  
            image = image.convert("RGB")

        # Convert PIL image to OpenCV format (BGR)
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Perform OCR on the image
        results = ImageProcessor._paddle_ocr.ocr(image_np, cls=True)

        logger.debug(f"OCR raw results: {results}")  # Debugging info

        # Handle None or empty case
        if results is None or all(not result for result in results):
            logger.warning("No text detected.")
            return 0

        # Extract detected angles
        angles = [line[1][1] for result in results if result for line in result]

        if not angles:
            logger.warning("No angles detected.")
            return 0

        detected_angle = sum(angles) / len(angles)
        logger.info(f"Detected rotation angle: {detected_angle}")

        return detected_angle

    @staticmethod
    def rotate_image(image: Image.Image,
                     method: Literal["paddle", "hough", "contour"] = "hough") -> Image.Image:
        """Rotate the image based on the detected angle using PaddleOCR."""

        if method == "hough":
            rotation_value = ImageProcessor.detect_rotation_hough(image)
        elif method == "paddle":
            rotation_value = ImageProcessor.detect_rotation_paddle(image)
        elif method == "contour":
            rotation_value = ImageProcessor.detect_rotation_contour(image)

        logger.info(f"Applying rotation: {rotation_value}°")

        if rotation_value == 0:
            return image  # No rotation needed

        # Convert PIL image to NumPy
        image_np = np.array(image)

        # Compute rotation matrix
        image_center = tuple(np.array(image_np.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, -rotation_value, 1.0)  # Negative for correct direction

        # Rotate the image
        rotated_np = cv2.warpAffine(image_np, rot_mat, image_np.shape[1::-1], flags=cv2.INTER_LINEAR)

        # Convert back to PIL image
        return Image.fromarray(rotated_np)
