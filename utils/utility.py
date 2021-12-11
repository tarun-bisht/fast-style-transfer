import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt


def load_image(image_path, dim=None, resize=False):
    img = Image.open(image_path)
    if dim:
        if resize:
            img = img.resize(dim)
        else:
            img.thumbnail(dim)
    img = img.convert("RGB")
    return np.array(img)


def load_url_image(url, dim=None, resize=False):
    img_request = requests.get(url)
    img = Image.open(BytesIO(img_request.content))
    if dim:
        if resize:
            img = img.resize(dim)
        else:
            img.thumbnail(dim)
    img = img.convert("RGB")
    return np.array(img)


def array_to_img(array):
    array = np.array(array, dtype=np.uint8)
    if np.ndim(array) > 3:
        assert array.shape[0] == 1
        array = array[0]
    return Image.fromarray(array)


def save_image(image, path):
    image.save(path)


def plot_images_grid(images, num_rows=1):
    n = len(images)
    if n > 1:
        num_cols = np.ceil(n / num_rows)
        fig, axes = plt.subplots(ncols=int(num_cols), nrows=int(num_rows))
        axes = axes.flatten()
        fig.set_size_inches((15, 15))
        for i, image in enumerate(images):
            axes[i].imshow(image)
    else:
        plt.figure(figsize=(10, 10))
        plt.imshow(images[0])
    plt.show()
