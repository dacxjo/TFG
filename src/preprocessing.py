import cv2
import numpy as np


def crop_image_roi(image: np.ndarray) -> np.ndarray:
    """
    Crop image ROI from given path
    :param image: Image to crop
    :return: Cropped image ROI
    """
    _, thresh = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply clahe to given image
    :param image: Image to apply clahe
    :return: Processed image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    return image

