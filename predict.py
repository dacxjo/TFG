import argparse

import numpy as np
import onnxruntime as rt
from PIL import Image

from config import SPLIT_TEST_DIR, MODELS_DIR
from src.modeling.transforms import get_transforms


def main():
    print("Starting inference...")
    parser = argparse.ArgumentParser(description="Run inference on a test image using an ONNX model.")
    parser.add_argument("--test_image", type=str, help="Path to the test image.")
    parser.add_argument("--model_path", type=str, help="Path to the ONNX model.")

    args = parser.parse_args()

    class_names = ['her2-enriched', 'luminal-a', 'luminal-b', 'triple-negative']

    if args.test_image:
        test_image = args.test_image
    else:
        test_image = SPLIT_TEST_DIR / 'triple-negative' / 'D2-0218_CC-L.png'

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = MODELS_DIR / 'artifacts' / 'best.onnx'

    session = rt.InferenceSession(model_path)

    input_name = session.get_inputs()[0].name

    image = Image.open(test_image).convert('RGB')
    image = np.array(image)

    # Pick the right transform
    test_transform = get_transforms(augment=False)
    transformed = test_transform(image=image)

    transformed_image = transformed['image']

    input_tensor = np.expand_dims(transformed_image, axis=0)

    result = session.run(None, {input_name: input_tensor})

    predicted_class = np.argmax(result[0], axis=1)

    print(f"Predicted class: {predicted_class[0]}")

    predicted_class_name = class_names[predicted_class[0]]
    print(f"Predicted class name: {predicted_class_name}")


if __name__ == "__main__":
    main()
