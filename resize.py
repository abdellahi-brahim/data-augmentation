import cv2
import numpy as np

sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

def get_closest(y):
    return min(sizes, key=lambda x: abs(x - y))

def center_crop(img, new_width=None, new_height=None):
    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img

def rectangle2square(image):
    width, height, _ = image.shape

    if width > height:
        image = center_crop(image, new_width=height)
    else:
        image = center_crop(image, new_height=width)

    return image

def resize2square(image, size, method=cv2.INTER_CUBIC):
    image = rectangle2square(image)
    return cv2.resize(image, dsize=(size, size), interpolation=method)
