from finast.utils import *
from PIL import Image
from pdf2image import convert_from_path

def test_cost_tracker():
    cost = CostTracker.track_cost(100, 200, "gpt-4o")
    print(cost)

def test_image_augmentor_rotation():
    image = convert_from_path("finast/tests/media/test.pdf")[0]
    augmented_image = ImageProcessor.rotate_image(image=image)
    augmented_image.show()

if __name__ == "__main__":
    test_cost_tracker()
    test_image_augmentor_rotation()
    print("All tests passed!")