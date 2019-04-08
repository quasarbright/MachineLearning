import os
from PIL import Image
flat_path = 'D:\\datasets\\animeface-character-dataset\\flat'
images = os.listdir(flat_path)
images = [os.path.join(flat_path, img) for img in images]
def get_width(img):
    img = Image.open(img).width
widths = list(map(lambda image: Image.open(image).width, images))
heights = list(map(lambda image: Image.open(image).height, images))
min_width = min(widths)
max_width = max(widths)
min_height = min(heights)
max_height = max(heights)
print(min_width, min_height)
print(max_width, max_height)
