import cv2
import numpy as np


def smart_crop(image: np.ndarray, additional_margin=0) -> np.ndarray:
    """
    Reduces the image to the bounding box of the non-zero pixels
    :param additional_margin: Additional margin to add to the bounding box
    :param image: Image to crop
    :return: Cropped image
    """
    _, thresh = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Ensure the cropping indices are within the image bounds
    y_start = max(y - additional_margin, 0)
    y_end = min(y + h + additional_margin, image.shape[0])
    x_start = max(x - additional_margin, 0)
    x_end = min(x + w + additional_margin, image.shape[1])

    cropped_image = image[y_start:y_end, x_start:x_end]

    return cropped_image


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply clahe to given image
    :param image: Image to apply clahe
    :return: Processed image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize image
    :param image: Image to normalize
    :return: Normalized image
    """
    result = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    result = result.astype(np.uint8)
    return result


def median_filter(image: np.ndarray, kernel_size=1) -> np.ndarray:
    """
    Apply median filter to image
    :param image: Image to apply median filter
    :param kernel_size: Kernel size
    :return: Processed image
    """
    return cv2.medianBlur(image, kernel_size)


def preprocess_image(image: np.ndarray, additional_margin=0, kernel_size=3) -> np.ndarray:
    """
    Preprocess image
    :param image: Image to preprocess
    :param additional_margin: Additional margin
    :param kernel_size: Kernel size
    :return: Processed image
    """
    cropped = smart_crop(image, additional_margin)
    median = median_filter(cropped, kernel_size)
    normalized = normalize(median)
    result = apply_clahe(normalized)
    return result
