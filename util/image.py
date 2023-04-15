import requests
import warnings

import numpy as np
from PIL import Image


IMAGE_DIM = 224


def load_image_from_url(url):
    # exclude images with any exceptions or warnings
    warnings.simplefilter("error")
    try:
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB').resize((IMAGE_DIM, IMAGE_DIM))
    except:
        image = None
    warnings.simplefilter("default")

    return image


def create_dummy_image():
    image = Image.new('RGB', (IMAGE_DIM, IMAGE_DIM))
    
    return image


def save_image(pixels):
    pixels = np.array(pixels)
    pixels = np.transpose(pixels, (1,2,0))
    new_image = Image.fromarray((pixels).astype(np.uint8))
    new_image.save('new.png')