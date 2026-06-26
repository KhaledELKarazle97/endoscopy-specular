import numpy as np
import cv2
from endoscopy_specular import detect_specular_reflection

def test_runs_without_crashing():
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    mask = detect_specular_reflection(img)

    assert mask is not None
    assert mask.shape == (256, 256)
    assert mask.dtype == np.uint8