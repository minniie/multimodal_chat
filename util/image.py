import requests
import warnings

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